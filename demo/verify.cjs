/* Verify the JS port against the Python original.
 * Usage: node verify.cjs   (run from the demo/ directory)
 * Checks: same top 10 and probabilities within 1e-6 on the full history,
 * and the three backtest splits reproduce the Python log scores.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { buildForecast, scoreForecast, baselineForecast } = require("./forecaster.js");

const demo = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const history = demo.history;
const pyForecast = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "out", "forecast.json"), "utf8"));

let fail = 0;
function check(name, ok, detail) {
  console.log((ok ? "PASS" : "FAIL") + "  " + name + (detail ? "  (" + detail + ")" : ""));
  if (!ok) fail++;
}

// 1. Full-history forecast: same top 10, same probabilities.
// Tolerance 1.1e-6: Python rounds to 6 decimals with banker's rounding,
// JS with half-up, so a rounding tie can differ by exactly 1e-6
// (2 of 228 rows do). The algorithm itself is identical.
const js = buildForecast(history, 72);
check("top10 identical", JSON.stringify(js.top10) === JSON.stringify(pyForecast.top10),
  "js=" + JSON.stringify(js.top10));
const pyMap = new Map(pyForecast.probabilities.map(r => [r.universe_id + ":" + r.place, r.p]));
let maxDiff = 0, missing = 0;
for (const r of js.probabilities) {
  const p = pyMap.get(r.universe_id + ":" + r.place);
  if (p === undefined) { missing++; continue; }
  maxDiff = Math.max(maxDiff, Math.abs(p - r.p));
}
check("probabilities within 1.1e-6", maxDiff <= 1.1e-6 && missing === 0,
  "max diff=" + maxDiff.toExponential(2) + ", missing=" + missing + ", js rows=" + js.probabilities.length + ", py rows=" + pyForecast.probabilities.length);

// 2. Backtest splits reproduce the Python numbers.
const expected = { "2026-11-04": -39.80, "2026-11-05": -39.48, "2026-11-06": -35.35 };
for (const cut of Object.keys(expected)) {
  const c = Date.parse(cut + "T00:00:00Z");
  const train = history.filter(r => Date.parse(r.at) < c);
  const test = history.filter(r => Date.parse(r.at) >= c);
  const s = scoreForecast(buildForecast(train, 72), test);
  const b = scoreForecast(baselineForecast(train), test);
  check("split " + cut + " score", Math.abs(s - expected[cut]) < 0.05,
    "js=" + s.toFixed(2) + " py=" + expected[cut] + " baseline=" + b.toFixed(2));
}

// 3. Full-history forecast scored on the dev reads (Nov 8-10).
const devScore = scoreForecast(js, demo.dev);
console.log("INFO  full-history forecast vs dev reads (Nov 8-10): " + devScore.toFixed(2));

process.exit(fail ? 1 : 0);
