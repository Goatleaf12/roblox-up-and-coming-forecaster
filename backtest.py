"""Backtest the forecaster on archive reads with the official log-score rule.

Train on reads before CUT, score on reads at/after CUT. Compares the
Gaussian rank-diffusion forecaster against a persistence baseline
(last chart carried forward).

Expected output (verified 2026-11-13):
  split Nov 4: forecaster -39.80 vs baseline -52.11
  split Nov 5: forecaster -39.48 vs baseline -55.85
  split Nov 6: forecaster -35.35 vs baseline -52.13
  mean:        forecaster -38.21 vs baseline -53.36
Higher (closer to zero) is better. Log scores are negative: 0 would be a
perfect prediction.
"""
import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecaster import build_forecast, parse_time  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TASK = {"window": {"slots": 72, "slot_minutes": 60}}
CUTS = ["2026-11-04T00:00:00Z", "2026-11-05T00:00:00Z", "2026-11-06T00:00:00Z"]


def load_rows():
    with open(os.path.join(DATA, "history.ndjson")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["at"])
    return rows


def ts(s):
    return parse_time(s)


def score_forecast(forecast, test_reads):
    """Mean log score of a forecast over test reads.

    forecast: {'probabilities': [{'universe_id', 'place', 'p'}], ...}.
    For each read and each place 1..10, take the probability assigned to
    the game that actually held that place; games the forecast never named
    share the unnamed mass (1 - named mass) evenly.
    """
    probs = {(r["universe_id"], r["place"]): r["p"]
             for r in forecast["probabilities"]}
    total = 0.0
    for r in test_reads:
        chart = r["up_and_coming"]
        by_rank = {g["rank"]: g["universe_id"] for g in chart}
        chart_uids = {g["universe_id"] for g in chart}
        s = 0.0
        for k in range(1, 11):
            uid = by_rank.get(k)
            if uid is None:
                continue
            named_sum = sum(p for (u, pl), p in probs.items() if pl == k)
            named_uids = {u for (u, pl) in probs if pl == k}
            if (uid, k) in probs:
                p = probs[(uid, k)]
            else:
                unnamed = 1.0 - named_sum
                n_unnamed = max(1, len(chart_uids - named_uids))
                p = unnamed / n_unnamed
            s += math.log(max(p, 0.0001))
        total += s
    return total / len(test_reads)


def baseline_forecast(train):
    """Persistence: last chart carried forward with fixed spreads."""
    last = train[-1]["up_and_coming"]
    by_rank = {g["rank"]: g["universe_id"] for g in last}
    probs = []
    for k in range(1, 11):
        probs.append({"universe_id": by_rank[k], "place": k, "p": 0.5})
        if k == 1:
            probs.append({"universe_id": by_rank[2], "place": 1, "p": 0.3})
        else:
            probs.append({"universe_id": by_rank[k - 1], "place": k, "p": 0.15})
            if k + 1 in by_rank:
                probs.append({"universe_id": by_rank[k + 1], "place": k, "p": 0.15})
    return {"probabilities": probs,
            "top10": [by_rank[k] for k in range(1, 11) if k in by_rank]}


def top10_hits(forecast, test_reads):
    tot = 0
    for r in test_reads:
        top = {g["universe_id"] for g in r["up_and_coming"][:10]}
        tot += len(set(forecast["top10"]) & top)
    return tot / len(test_reads)


def main():
    rows = load_rows()
    print(f"history: {len(rows)} reads, {rows[0]['at']} .. {rows[-1]['at']}")
    f_scores, b_scores = [], []
    for cut in CUTS:
        c = ts(cut)
        train = [r for r in rows if ts(r["at"]) < c]
        test = [r for r in rows if ts(r["at"]) >= c]
        f = build_forecast(train, TASK)
        b = baseline_forecast(train)
        fs, bs = score_forecast(f, test), score_forecast(b, test)
        f_scores.append(fs)
        b_scores.append(bs)
        print(f"split {cut[:10]}: forecaster {fs:6.2f} (top10 hits "
              f"{top10_hits(f, test):.2f}/read) vs baseline {bs:6.2f} "
              f"-> gain {fs - bs:+.2f}")
    print(f"mean: forecaster {sum(f_scores) / len(f_scores):.2f} vs "
          f"baseline {sum(b_scores) / len(b_scores):.2f}")


if __name__ == "__main__":
    main()
