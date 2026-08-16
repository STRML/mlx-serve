// website_benchmarks_logic.mjs — unit-tests the pure logic embedded in
// website/benchmarks/index.html: row validation, RTDB payload parsing, median,
// cell grouping, cross-session ratio aggregation and filtering.
// Invoked by test_website_pages.sh when node is available; exits non-zero on
// the first failed assertion.
//
// Like website_tier_list_logic.mjs it evals the page's module script up to the
// DOM-dependent rendering section, so the code under test is the exact code the
// browser runs — no copies.
//
// The grouping logic here MIRRORS BenchmarkStore.cellKey/aggregate in the Swift
// app. If the two diverge, the app and the website quote different numbers for
// the same data, so both sides are tested and the duplication is deliberate.
import { readFileSync } from "node:fs";

const html = readFileSync("website/benchmarks/index.html", "utf8");
const script = html.split('<script type="module">')[1]?.split("</script>")[0];
if (!script) { console.error("ASSERT FAIL: module script not found in page"); process.exit(1); }
const pure = script.split("// ── rendering")[0];
if (pure.length === script.length) { console.error("ASSERT FAIL: rendering marker missing"); process.exit(1); }

const asserts = `
function assert(c, m) { if (!c) { console.error("ASSERT FAIL: " + m); process.exit(1); } }

function row(over) {
  return Object.assign({
    sessionId: "s1", suiteId: "standard-v1", armId: "defaults", armLabel: "Defaults",
    modelId: "mlx-community/Qwen3.6-27B-4bit", isLossy: false,
    prefillTps: 900, decodeTps: 50,
    hardware: { chip: "Apple M4 Max", gpuCores: 40, ramGB: 128, osVersion: "27.0", onBattery: false },
  }, over || {});
}

// ── row validation: an open database means anything can arrive ─────────────
assert(isValidRow(row()), "a well-formed row validates");
assert(!isValidRow(null), "null is not a row");
assert(!isValidRow({}), "an empty object is not a row");
assert(!isValidRow(row({ decodeTps: 0 })), "zero decode is not a measurement");
assert(!isValidRow(row({ decodeTps: -5 })), "negative decode is rejected");
assert(!isValidRow(row({ decodeTps: "fast" })), "a string decode is rejected");
assert(!isValidRow(row({ hardware: null })), "a row with no hardware is unfilterable");
assert(!isValidRow(row({ hardware: { chip: "", ramGB: 8 } })), "a blank chip is unfilterable");
assert(!isValidRow(row({ modelId: "" })), "a blank model is unusable");

// ── RTDB payload shape: an OBJECT keyed by push id, or null when empty ─────
assert(parseRows(null).length === 0, "an empty database renders as no rows");
assert(parseRows({}).length === 0, "an empty object yields no rows");
assert(parseRows({ "-Na": row(), "-Nb": row() }).length === 2, "push-keyed object yields its rows");
// One junk write must never blank the board.
assert(parseRows({ "-Na": row(), "-Njunk": { nonsense: true } }).length === 1,
       "a malformed row is skipped, the good one survives");

// ── median ────────────────────────────────────────────────────────────────
assert(median([]) === 0, "median of nothing is 0");
assert(median([42]) === 42, "median of one");
assert(median([10, 20]) === 15, "median of two averages");
assert(median([50, 51, 20]) === 50, "one slow run does not drag the median");

// ── cell grouping: only genuinely comparable rows share a median ───────────
const m4max40 = row();
const m4max32 = row({ hardware: { chip: "Apple M4 Max", gpuCores: 32, ramGB: 128 } });
assert(cellKey(m4max40) !== cellKey(m4max32),
       "GPU core count separates cells — 32 and 40 core M4 Max are different machines");
assert(cellKey(row({ decodeTps: 99 })) === cellKey(row()),
       "the measurement itself is not part of the key");
assert(cellKey(row({ modelId: "other" })) !== cellKey(row()), "model separates cells");
assert(cellKey(row({ armId: "pld" })) !== cellKey(row()), "arm separates cells");
// Engine version is deliberately NOT in the key: fragmenting by release would
// leave every cell at n=1 forever.
assert(cellKey(row({ engineVersion: "26.9.0" })) === cellKey(row({ engineVersion: "26.8.1" })),
       "engine version does not fragment cells");

// ── aggregate ─────────────────────────────────────────────────────────────
const agg = aggregate([row({ decodeTps: 40 }), row({ decodeTps: 50 }), row({ decodeTps: 60 })]);
assert(agg.length === 1, "identical machines collapse into one cell");
assert(agg[0].decodeTps === 50, "cell reports the median");
assert(agg[0].sampleCount === 3, "cell reports how many results it came from");

const split = aggregate([row(), row({ armId: "kv-quant-4", isLossy: true })]);
assert(split.length === 2, "different arms never share a median");
assert(aggregate([row({ decodeTps: 10 }), row({ armId: "pld", decodeTps: 90 })])[0].armId === "pld",
       "cells sort fastest first");

// ── ratio view: the only comparison valid across machines ─────────────────
const oneSession = [
  row({ sessionId: "a", armId: "defaults", decodeTps: 50 }),
  row({ sessionId: "a", armId: "pld", armLabel: "PLD on", decodeTps: 60 }),
];
const r1 = ratiosBySession(oneSession);
assert(r1.length === 1 && r1[0].armId === "pld", "the non-anchor arm is reported");
assert(Math.abs(r1[0].ratio - 1.2) < 1e-9, "ratio is against the session's own defaults");

// A session with no anchor contributes nothing — comparing it against someone
// else's baseline would publish a different quantity under the same name.
assert(ratiosBySession([row({ sessionId: "b", armId: "pld", decodeTps: 60 })]).length === 0,
       "an anchorless session is dropped");
assert(ratiosBySession([
  row({ sessionId: "c", armId: "defaults", decodeTps: 0 }),
  row({ sessionId: "c", armId: "pld", decodeTps: 60 }),
]).length === 0, "a zero anchor never produces an infinite ratio");

// Ratios median ACROSS sessions, so a fast Mac cannot outvote a slow one.
const twoSessions = [
  row({ sessionId: "a", armId: "defaults", decodeTps: 10 }),
  row({ sessionId: "a", armId: "pld", decodeTps: 20 }),          // 2.0x on a slow Mac
  row({ sessionId: "b", armId: "defaults", decodeTps: 100 }),
  row({ sessionId: "b", armId: "pld", decodeTps: 140 }),         // 1.4x on a fast Mac
];
const r2 = ratiosBySession(twoSessions);
assert(r2.length === 1 && r2[0].sampleCount === 2, "both sessions counted once each");
assert(Math.abs(r2[0].ratio - 1.7) < 1e-9, "ratios median across sessions, not weighted by speed");

// ── filters ───────────────────────────────────────────────────────────────
const mixed = [
  row({ hardware: { chip: "Apple M4 Max", gpuCores: 40, ramGB: 128 } }),
  row({ hardware: { chip: "Apple M4", gpuCores: 10, ramGB: 16 } }),
  row({ armId: "kv-quant-4", isLossy: true }),
];
assert(applyFilters(mixed, { chip: "Apple M4" }).length === 1, "chip filter");
assert(applyFilters(mixed, { ram: "16" }).length === 1, "memory filter");
assert(applyFilters(mixed, { quality: "lossless" }).length === 2, "lossless filter drops lossy rows");
assert(applyFilters(mixed, {}).length === 3, "no filters keeps everything");
assert(applyFilters(mixed, { model: "nope" }).length === 0, "unknown model matches nothing");

// ── the fetch URL must actually be RTDB's query grammar ───────────────────
const url = fetchURL();
assert(url.includes("%22%24key%22"), 'orderBy="$key" must be URL-encoded or RTDB 400s');
assert(url.includes("limitToLast="), "fetch is bounded, never the whole database");

console.log("website benchmarks logic: all assertions passed");
`;

const module = pure + asserts;
try {
  // eslint-disable-next-line no-new-func
  new Function(module)();
} catch (e) {
  console.error("ASSERT FAIL: " + (e && e.message ? e.message : String(e)));
  process.exit(1);
}
