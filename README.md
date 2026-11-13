# Roblox Up-and-Coming Forecaster

An agent that forecasts the **top 10 of Roblox's Up-and-Coming chart** (computer,
US) over the next 72 hours — which games will break into the top 10, and with
what probability for each place.

**Live demo:** https://roblox-forecaster.pages.dev — pick a training cut, run the
forecast in your browser, and see the backtest score against the reads that
actually followed.

## How it works

Each game's rank is treated as a drifting, diffusing random walk:

- **Drift by rank bucket.** The expected rank change per 6 hours is estimated
  from every pair of consecutive chart reads, separately for rank buckets
  1–3, 4–10, 11–20, 21–40, 41–200. Top games cool off; mid-chart games climb.
- **Own velocity.** The drift is blended 70/30 with the game's own rank
  velocity over the last 40 hours (clamped to ±15 ranks per 6h, so one noisy
  jump cannot dominate).
- **Diffusion.** Uncertainty widens with the horizon (σ² = 2.0² + 0.7·hours).
  Per-place probabilities are the time-average of each game's Gaussian mass
  on that place over the 72-hour window.
- **Unnamed mass.** 30% of every place is left unnamed: about a sixth of
  top-10 slots in the archive go to games never seen before.
- **Eligibility taper.** Games whose 28-day Up-and-Coming eligibility ends
  inside the window are tapered by the fraction of the window they survive.

Deterministic: no randomness, sorted iteration, fixed float rounding — two
runs write byte-identical output. Standard library only.

## Backtest evidence

Mean log score over the reads after each training cut (higher — closer to
zero — is better; 0 would be a perfect prediction). Reproduce with
`python3 backtest.py`.

| Train before | Forecaster | Persistence baseline | Gain |
|---|---|---|---|
| Nov 4, 2026 | −39.80 | −52.11 | +12.31 |
| Nov 5, 2026 | −39.48 | −55.85 | +16.37 |
| Nov 6, 2026 | −35.35 | −52.13 | +16.78 |
| **Mean** | **−38.21** | **−53.36** | **+15.15** |

The forecaster also beats the baseline on a 72-hour-window protocol over five
cuts (Nov 2–6, mean −37.57 vs −38.53), and scores −36.23 on the held-out dev
reads of Nov 8–10 (baseline −43.51). The top-10 hit rate is 4.6–6.0 games per
read out of 10.

## Run it

```sh
# Forecast the 72h window after the last read in data/
python3 forecaster.py            # writes out/forecast.json

# Reproduce the backtest table above
python3 backtest.py

# Verify the JavaScript port used by the demo matches the Python agent
cd demo && node verify.cjs
```

`forecaster.py` reads `data/history.ndjson` (34 chart reads, Oct 30 – Nov 7,
2026) and `data/task.json` (72 hourly slots). Override with
`FORECASTER_INPUT_DIR` / `FORECASTER_OUTPUT_DIR`.

## Demo

`demo/` is a dependency-free static page: it loads `data.json` (the same
reads, trimmed to the fields the model needs) and runs a faithful JavaScript
port of `forecaster.py` in the browser. The port is verified against the
Python original — same top 10, probabilities within 1.1e-6 (the only
differences are rounding ties at the 6th decimal). Serve it with any static
host, e.g. `python3 -m http.server -d demo`.

## Data

- `data/history.ndjson` — 34 reads of Roblox's Up-and-Coming chart (plus Top
  Trending / Top Playing Now where read), Oct 30 – Nov 7, 2026, from Roblox's
  public chart API via make.fun's recorder and the public archive below.
- `data/dev_reads.ndjson` — 10 held-out reads, Nov 8–10, 2026, reconstructed
  from the same public archive.
- Attribution and the archive's MIT licence: `data/NOTICE.txt`. Chart data
  comes from the MIT-licensed
  [Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots)
  archive.

## Licence

MIT — see `LICENSE`. Built as an entry for zkVerify Agent Games (Wildcard
track): an agent that is useful on its own, with a working public demo — see
`USE_CASE.md`.
