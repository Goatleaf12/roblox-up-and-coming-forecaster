# Use case — Roblox Up-and-Coming Forecaster (Wildcard)

## The problem

Roblox's Up-and-Coming chart is where new games break out — but it turns over
fast. A game at rank 40 today can be top 10 tomorrow, and a game at rank 3
can slide out just as quickly. Developers, curators, and players who watch the
chart have no principled way to answer "which games will be top 10 in three
days?": the chart itself only shows where games are *now*, and naive
"yesterday's top 10 stays top 10" is wrong most of the time (in our archive,
persistence gets only ~4–5 of the next top 10 right per read, and scores
−52 to −56 in log score).

## Who it is for

- **Game developers** deciding where to spend their launch and update effort:
  a rising game is worth watching (or collaborating with) before it peaks.
- **Curators and community managers** who surface new games: a ranked,
  probabilistic shortlist beats a gut-feel list.
- **Players** who want to find tomorrow's popular games today.

## What the agent does

Given a history of chart reads, it outputs a full probabilistic forecast for
the next 72 hours: for each of the 10 places, a ranked list of games with
probabilities, plus a single top-10 shortlist. The model is a Gaussian
rank-diffusion: per-bucket drift estimated from the archive, blended with
each game's own 40-hour rank velocity, with diffusion widening over the
horizon, an unnamed share for brand-new games, and a taper for games whose
28-day chart eligibility expires mid-window. It is deterministic and
reproducible — the same reads always give the same forecast.

## Evidence it works

Backtested on three archive splits (train before Nov 4 / 5 / 6, scored on the
reads after): mean log score **−38.21 vs −53.36** for the persistence baseline
(+15.15 points), winning on every split, and −36.23 on held-out reads from
Nov 8–10 (baseline −43.51). Reproduce with `python3 backtest.py`.

## How to try it

- **Live demo:** https://roblox-forecaster.pages.dev — pick a training cut,
  run the forecast in the browser, see the top 10, the per-place
  probabilities, and the backtest score against what actually happened.
- **Repo:** https://github.com/Goatleaf12/roblox-up-and-coming-forecaster —
  `forecaster.py` (the agent, stdlib only), `backtest.py`, the archive data,
  and the demo (a verified JavaScript port of the same algorithm).

## Honest limitations

- The bundled data ends Nov 10, 2026: the demo forecasts from archive reads,
  not from the live chart. Pointed at fresh reads (same NDJSON schema) the
  agent forecasts the live chart unchanged — but it cannot fetch Roblox data
  itself; a live deployment would need a chart reader feeding it new reads.
- The model only sees chart ranks and game ages. It does not see update
  announcements, social buzz, or player counts — a surprise update can move a
  game faster than any rank history predicts. The unnamed 30% is the model's
  explicit admission of that.
- It forecasts the Up-and-Coming chart (computer, US) only; other charts and
  regions have different dynamics.
