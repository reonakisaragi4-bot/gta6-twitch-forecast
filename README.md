# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/reonakisaragi4-bot/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/reonakisaragi4-bot/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 19 November 2026 (UTC), from the first hour's
viewer count and the release-day curves of past big game launches. For every
hour 2–24 it writes five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent viewers, scored by pinball loss on the log scale.

How it works: at run time the program fits, for each hour, a quadratic
regression of log viewers on log hour-1 viewers (plus log hour-1 channels)
across the comparables in the case file, and blends it half-and-half with a
kernel-weighted nearest-neighbour forecast of the same hour's log-ratio
curve. Launches whose first hour is already huge stay flat or decay, while
launches whose first hour is tiny ramp up toward the evening peak — the
quadratic bends with that, and the neighbours keep the forecast honest
where the regression smooths over. Deterministic, standard library only,
no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `<id>.in` in the input directory and writes one
`<id>.out` per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cdrpe06y&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
