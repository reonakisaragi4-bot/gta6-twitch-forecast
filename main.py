"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

The program reads one case (makefun/twitch-ccu-case/v1) and writes, for each
of hours 2..24 of the release day, five quantiles of that hour's concurrent
Twitch viewers in the game's category.

Method
------
The case carries two things that matter: hour 1's print (the anchor) and the
release-day curves of dozens of past big launches ("comparables"). Two
forecasters are built from those comparables at run time and blended:

1. A quadratic regression. For each hour h, log viewers are regressed on
   log hour-1 viewers, its square, and log hour-1 channels across the
   comparables. The square term matters: launches whose hour 1 is already
   huge (a simultaneous global unlock) stay flat or decay, while launches
   whose hour 1 is tiny (only New Zealand and Asia unlocked yet) ramp up
   several-fold toward the evening peak. A straight line splits the
   difference and is wrong at both ends; the quadratic bends with it.
   The spread around the fit is the residual standard deviation of that
   hour, floored, and the five quantiles are the fitted mean plus the
   normal quantiles times that spread.

2. A nearest-neighbour forecast. The comparables closest to the case in
   log hour-1 viewers lend their own log-ratio curves (hour h over hour 1),
   kernel-weighted, and the forecast's quantiles are the weighted quantiles
   of those ratios, re-anchored on this case's hour-1 print. This keeps
   the forecast honest when the target sits in a regime the regression
   smooths over.

The final forecast averages the two quantile sets in log space, half and
half. If the comparables are too few or a fit fails, the program falls
back to a persistence forecast (hour 1's print, spread widening like a
random walk in log space), so it always writes a valid file.

Everything is deterministic: no randomness, no network, standard library
only. Two runs on the same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing beside
# this file is imported, so that does not matter.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Neighbours lent by the kernel forecast, and the kernel's log bandwidth.
KNN_K = 20
KNN_SIGMA = 1.2
# Blend weight of the regression forecast (the rest is the neighbour one).
BLEND_REGRESSION = 0.5
# The regression's spread: residual sd, floored, then shrunk a little.
SPREAD_FLOOR = 0.15
SPREAD_SHRINK = 0.85
# Centring constants for the regression's features (keeps the normal
# equations well conditioned; the fit is unchanged by the shift).
X_CENTRE = 10.0
CH_CENTRE = 7.0
# Fallback persistence spread, in log units per square-root hour.
PERSIST_SIGMA = 0.15
# A comparable needs this many release-day hours to join the fit.
MIN_TRAIN = 5


def parse_time(text):
    """An ISO 8601 UTC timestamp as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day(comp):
    """A comparable's release-day curve: 24 (viewers, channels) pairs.

    Returns None when the comparable has no complete release day (the
    GTA VI category's own pre-release entry, for instance, only covers
    its premiere day) or its hour 1 holds no viewers to anchor on.
    """
    start = parse_time(comp["day_start"])
    day = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            slot = int((parse_time(point["from"]) - start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= slot < 24:
            day[slot] = (viewers, point.get("channels"))
    if len(day) < 24:
        return None
    if day[0][0] < 1:
        return None
    return [day[h] for h in range(24)]


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns None when the system is singular.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
        out[i] = total / aug[i][i]
    return out


def fit_hour(train, hour, target_x, target_ch):
    """Regression forecast of one hour: (median log viewers, log spread).

    Fits log viewers at `hour` on [1, u, u^2, w] across the training
    curves, where u is centred log hour-1 viewers and w centred log
    hour-1 channels. Returns None when the fit fails.
    """
    rows = []
    answers = []
    slot = hour - 1  # `hour` is 1-based; the curve is a 0-based list
    for curve in train:
        v1, ch1 = curve[0]
        u = math.log(max(v1, 1)) - X_CENTRE
        w = math.log(max(ch1 or 1, 1)) - CH_CENTRE
        rows.append([1.0, u, u * u, w])
        answers.append(math.log(max(curve[slot][0], 1)))
    width = len(rows[0])
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(width)] for i in range(width)]
    rhs = [sum(r[i] * y for r, y in zip(rows, answers)) for i in range(width)]
    coef = solve_linear(normal, rhs)
    if coef is None:
        return None
    resid = [y - sum(c * x for c, x in zip(coef, r)) for r, y in zip(rows, answers)]
    mean_resid = sum(resid) / len(resid)
    var = sum((e - mean_resid) ** 2 for e in resid) / max(len(resid) - 1, 1)
    spread = max(math.sqrt(var), SPREAD_FLOOR) * SPREAD_SHRINK
    u = target_x - X_CENTRE
    w = target_ch - CH_CENTRE
    centre = coef[0] + coef[1] * u + coef[2] * u * u + coef[3] * w
    return centre, spread


def weighted_quantiles(pairs, levels):
    """Weighted quantiles of (value, weight) pairs, in increasing order."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        mid = ordered[len(ordered) // 2][0]
        return [mid] * len(levels)
    running = 0.0
    cumulative = []
    for value, weight in ordered:
        running += weight
        cumulative.append((value, running))
    out = []
    for level in levels:
        mark = level * total
        picked = cumulative[-1][0]
        for value, reached in cumulative:
            if reached >= mark:
                picked = value
                break
        out.append(picked)
    return out


def neighbour_quantiles(train, hour, target_log_v1):
    """Kernel-weighted quantiles of the log-ratio curve at one hour."""
    scored = []
    for curve in train:
        v1 = curve[0][0]
        dist = abs(math.log(max(v1, 1)) - target_log_v1)
        scored.append((dist, curve))
    scored.sort(key=lambda item: item[0])
    pairs = []
    slot = hour - 1  # `hour` is 1-based; the curve is a 0-based list
    for dist, curve in scored[:KNN_K]:
        weight = math.exp(-0.5 * (dist / KNN_SIGMA) ** 2)
        ratio = math.log(max(curve[slot][0], 1)) - math.log(max(curve[0][0], 1))
        pairs.append((ratio, weight))
    return weighted_quantiles(pairs, LEVELS)


def persistence_quantiles(p1, hour):
    """The fallback: hour 1's print, spread widening over the day."""
    spread = PERSIST_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def round2(value):
    """A number rounded to 2 decimals, half up; whole numbers as ints."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def forecast_case(case, name):
    """The forecast object for one case: {"hours": [...23 rows...]}."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is not None:
        p1 = max(1, first.get("viewers") or 1)
        ch1 = first.get("channels") or 1
    else:
        history = case.get("history") or []
        p1 = max(1, (history[-1].get("viewers") or 1) if history else 1)
        ch1 = 1
    target_log_v1 = math.log(p1)
    target_log_ch = math.log(max(ch1, 1))

    train = []
    for comp in case.get("comparables") or []:
        curve = release_day(comp)
        if curve is not None:
            train.append(curve)

    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        if len(train) < MIN_TRAIN:
            quantiles = persistence_quantiles(p1, hour)
        else:
            fitted = fit_hour(train, hour, target_log_v1, target_log_ch)
            neighbour = neighbour_quantiles(train, hour, target_log_v1)
            if fitted is None:
                quantiles = [p1 * math.exp(q) for q in neighbour]
            else:
                centre, spread = fitted
                regression = [math.exp(centre + z * spread) for z in Z]
                anchored = [p1 * math.exp(q) for q in neighbour]
                quantiles = [
                    math.exp(
                        BLEND_REGRESSION * math.log(regression[j])
                        + (1.0 - BLEND_REGRESSION) * math.log(anchored[j])
                    )
                    for j in range(len(LEVELS))
                ]
        # Keep the row ordered, inside the contest's bounds, 2 decimals.
        clean = []
        floor = 1.0
        for value in quantiles:
            value = min(max(value, 1.0), 1e9)
            value = max(value, floor)
            floor = value
            clean.append(round2(value))
        hours.append({"hour": hour, "q": clean})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
