/* Faithful JavaScript port of ../forecaster.py (same constants, same math).
 *
 * Model: per-game Gaussian rank diffusion. Each game's rank is a random
 * walk with (a) a drift that depends on its current rank bucket, estimated
 * from every pair of consecutive reads, blended 70/30 with (b) the game's
 * own recent rank velocity over the last 40 hours, and (c) diffusion that
 * widens with the forecast horizon. Per-place probabilities are the
 * time-average over the 72h window of each game's Gaussian mass on that
 * place. 30% of every place is left unnamed (new games); games whose
 * 28-day eligibility ends inside the window are tapered.
 *
 * Verified against the Python original: same inputs give the same top 10
 * and probabilities within 1.1e-6 (see verify.cjs). The only differences
 * are rounding ties at the 6th decimal (Python banker's rounding vs JS
 * half-up): 2 of 228 rows differ by exactly 1e-6.
 */
"use strict";

const SIGMA0 = 2.0;        // rank std at horizon 0
const DIFF_PER_H = 0.7;    // variance growth per forecast hour
const UNNAMED = 0.30;      // share of each place left unnamed
const MIN_P = 0.002;       // smallest probability row written
const MAX_RANK = 60;       // deepest last rank still modelled
const VEL_WEIGHT = 0.3;    // blend of own recent velocity into drift
const VEL_WINDOW_H = 40.0;
const VEL_CLAMP = 15.0;    // max |rank change| per 6h credited to velocity
const MAX_AGE_DAYS = 28.0;
const BUCKETS = [[1, 3], [4, 10], [11, 20], [21, 40], [41, 200]];

function erf(x) {
  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7.
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function normCdf(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function parseTime(s) { return Date.parse(s); } // ms since epoch
function round6(x) { return Math.round(x * 1e6) / 1e6; }

function driftTable(reads) {
  const table = new Map(); // "lo-hi" -> [mean, std]
  for (const [lo, hi] of BUCKETS) {
    const ds = [];
    for (let i = 0; i + 1 < reads.length; i++) {
      const a = reads[i], b = reads[i + 1];
      const dt = Math.max((parseTime(b.at) - parseTime(a.at)) / 3600000, 1.0);
      const am = new Map(a.up_and_coming.map(g => [g.universe_id, g.rank]));
      for (const g of b.up_and_coming) {
        if (am.has(g.universe_id) && am.get(g.universe_id) >= lo && am.get(g.universe_id) <= hi) {
          ds.push((g.rank - am.get(g.universe_id)) * (6.0 / Math.max(dt, 1.0)));
        }
      }
    }
    if (ds.length) {
      const mean = ds.reduce((s, d) => s + d, 0) / ds.length;
      const variance = ds.reduce((s, d) => s + (d - mean) * (d - mean), 0) / ds.length;
      table.set(lo + "-" + hi, [mean, Math.sqrt(variance)]);
    }
  }
  return table;
}

function driftForRank(table, rank) {
  for (const [lo, hi] of BUCKETS) {
    const v = table.get(lo + "-" + hi);
    if (v && rank >= lo && rank <= hi) return v[0];
  }
  return 0.0;
}

function buildForecast(reads, horizonH) {
  const endTime = parseTime(reads[reads.length - 1].at);
  const table = driftTable(reads);

  const lastRank = new Map(), lastSeen = new Map(), firstSeen = new Map(),
        created = new Map(), rankHist = new Map();
  for (const r of reads) {
    const t = parseTime(r.at);
    for (const g of r.up_and_coming) {
      const uid = g.universe_id;
      lastRank.set(uid, g.rank);
      lastSeen.set(uid, t);
      if (!firstSeen.has(uid)) firstSeen.set(uid, t);
      if (!rankHist.has(uid)) rankHist.set(uid, []);
      rankHist.get(uid).push([t, g.rank]);
    }
    for (const [uidStr, d] of Object.entries(r.details || {})) {
      if (d && d.created) {
        const c = Date.parse(d.created);
        if (!Number.isNaN(c)) created.set(Number(uidStr), c);
      }
    }
  }

  const horizons = [];
  for (let h = 0; h <= horizonH; h += 6) horizons.push(h);
  if (horizons[horizons.length - 1] < horizonH) horizons.push(horizonH);

  const games = [...lastRank.keys()]
    .filter(uid => lastRank.get(uid) <= MAX_RANK &&
      (endTime - lastSeen.get(uid)) < 48 * 3600 * 1000)
    .sort((a, b) => a - b);

  const windowEnd = endTime + horizonH * 3600 * 1000;
  const survive = new Map();
  for (const uid of games) {
    const born = created.get(uid) ?? firstSeen.get(uid);
    if (born === undefined) { survive.set(uid, 1.0); continue; }
    const expiry = born + MAX_AGE_DAYS * 86400 * 1000;
    if (expiry <= endTime) survive.set(uid, 0.0);
    else if (expiry >= windowEnd) survive.set(uid, 1.0);
    else {
      const frac = (expiry - endTime) / (windowEnd - endTime);
      survive.set(uid, 0.15 + 0.85 * frac);
    }
  }

  const probabilities = [];
  const member = new Map(), expectedPlace = new Map();
  for (let place = 1; place <= 10; place++) {
    const col = new Map();
    for (const uid of games) {
      const r0 = lastRank.get(uid);
      let drift = driftForRank(table, r0);
      const hist = (rankHist.get(uid) || [])
        .filter(([t]) => (endTime - t) <= VEL_WINDOW_H * 3600 * 1000);
      if (hist.length >= 2) {
        const [t0, rOld] = hist[0], [t1, rNew] = hist[hist.length - 1];
        const dt = Math.max((t1 - t0) / 3600000, 1.0);
        let velocity = (rNew - rOld) * (6.0 / dt);
        velocity = Math.max(-VEL_CLAMP, Math.min(VEL_CLAMP, velocity));
        drift = drift * (1.0 - VEL_WEIGHT) + velocity * VEL_WEIGHT;
      }
      let total = 0.0;
      for (const hh of horizons) {
        let mu = r0 + drift * (hh / 6.0);
        mu = Math.max(1.0, Math.min(110.0, mu));
        const sig = Math.sqrt(SIGMA0 * SIGMA0 + DIFF_PER_H * hh);
        total += normCdf((place + 0.5 - mu) / sig) - normCdf((place - 0.5 - mu) / sig);
      }
      col.set(uid, (total / horizons.length) * survive.get(uid));
    }
    const tot = [...col.values()].reduce((s, v) => s + v, 0);
    if (tot <= 0) continue;
    const ranked = [...col.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 100);
    for (const [uid, p] of ranked) {
      let pp = (p / tot) * (1.0 - UNNAMED);
      if (pp >= MIN_P) {
        pp = round6(pp);
        probabilities.push({ universe_id: uid, place, p: pp });
        member.set(uid, (member.get(uid) || 0) + pp);
        expectedPlace.set(uid, (expectedPlace.get(uid) || 0) + pp * place);
      }
    }
  }

  const order = [...member.keys()].sort((a, b) =>
    member.get(b) - member.get(a) ||
    (expectedPlace.get(a) / Math.max(member.get(a), 1e-9)) -
    (expectedPlace.get(b) / Math.max(member.get(b), 1e-9)) || a - b);
  const top10 = order.slice(0, 10);
  if (top10.length < 10) {
    for (const g of reads[reads.length - 1].up_and_coming) {
      if (!top10.includes(g.universe_id)) top10.push(g.universe_id);
      if (top10.length === 10) break;
    }
  }
  return { probabilities, top10 };
}

/* Mean log score of a forecast over test reads (same rule as backtest.py).
 * forecast: {probabilities: [{universe_id, place, p}], top10}.
 * Higher (closer to zero) is better; 0 would be perfect. */
function scoreForecast(forecast, testReads) {
  const probs = new Map(forecast.probabilities.map(r => [r.universe_id + ":" + r.place, r.p]));
  let total = 0.0;
  for (const r of testReads) {
    const byRank = new Map(r.up_and_coming.map(g => [g.rank, g.universe_id]));
    const chartUids = new Set(r.up_and_coming.map(g => g.universe_id));
    let s = 0.0;
    for (let k = 1; k <= 10; k++) {
      const uid = byRank.get(k);
      if (uid === undefined) continue;
      let namedSum = 0.0;
      const namedUids = new Set();
      for (const [key, p] of probs) {
        const [u, pl] = key.split(":");
        if (Number(pl) === k) { namedSum += p; namedUids.add(Number(u)); }
      }
      let p;
      if (probs.has(uid + ":" + k)) p = probs.get(uid + ":" + k);
      else {
        const unnamed = 1.0 - namedSum;
        const nUnnamed = Math.max(1, [...chartUids].filter(u => !namedUids.has(u)).length);
        p = unnamed / nUnnamed;
      }
      s += Math.log(Math.max(p, 0.0001));
    }
    total += s;
  }
  return total / testReads.length;
}

function baselineForecast(train) {
  const last = train[train.length - 1].up_and_coming;
  const byRank = new Map(last.map(g => [g.rank, g.universe_id]));
  const probs = [];
  for (let k = 1; k <= 10; k++) {
    probs.push({ universe_id: byRank.get(k), place: k, p: 0.5 });
    if (k === 1) probs.push({ universe_id: byRank.get(2), place: 1, p: 0.3 });
    else {
      probs.push({ universe_id: byRank.get(k - 1), place: k, p: 0.15 });
      if (byRank.has(k + 1)) probs.push({ universe_id: byRank.get(k + 1), place: k, p: 0.15 });
    }
  }
  return { probabilities: probs, top10: [...Array(10)].map((_, i) => byRank.get(i + 1)).filter(Boolean) };
}

if (typeof module !== "undefined") {
  module.exports = { buildForecast, scoreForecast, baselineForecast };
}
