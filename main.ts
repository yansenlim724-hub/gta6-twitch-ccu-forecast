// Forecast for gta6-twitch-ccu-forecast-01.
//
// Idea, in one paragraph: launch-day Twitch viewing follows the same daily
// shape across big console launches, but the *level* of the first hour tells
// you how much of that shape is still to come. A launch whose first hour is
// already huge is near its peak; one whose first hour is small has a long
// ramp ahead. So for each hour h I estimate two things from the past
// launches shipped inside the case file and average them in log space:
//   (a) a persistence forecast: hour 1's print times the median
//       log-ratio log(viewers_h / viewers_1) across similar past launches;
//   (b) a level regression: log(viewers_h) regressed on log(viewers_1),
//       which captures that the peak level varies far less across launches
//       than the first-hour level does.
// The spread comes from the interquartile range of those same log-ratios,
// shrunk by a factor (the regression already explains part of the spread).
// Everything is computed from the case file at run time; nothing is
// hard-coded from the practice data, so the same program works on the final
// case. Standard library only, no network, fully deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// The five quantile levels, in the order every output row lists them.
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Standard normal quantiles at those levels.
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
// How much of the log-ratio spread to keep (the regression already absorbs
// part of the cross-launch variation, so the raw spread is too wide).
const SPREAD_KEEP = 0.6;
// Weight of the level regression in the blended log-median.
const REGRESSION_WEIGHT = 0.5;
// Fallback spread for the persistence baseline, in log units per sqrt hour.
const BASELINE_SIGMA = 0.15;

type HourlyPoint = { from: string; viewers: number; channels?: number | null };

type Comparable = {
  name: string;
  day_start: string;
  unlock?: string | null;
  platforms?: string | null;
  hourly: HourlyPoint[];
};

type CaseFile = {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
};

/** Round to 2 decimals, half up, the way the contest's example does. */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** Linear-interpolated quantile of an already sorted array. */
function quantileOf(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/** Ordinary least squares of ys on xs; returns [intercept, slope, residSd]. */
function fitLine(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    rss += r * r;
  }
  const sd = n > 2 ? Math.sqrt(rss / (n - 2)) : 0.5;
  return [intercept, slope, Math.max(sd, 0.05)];
}

/**
 * Is this past launch a usable analogue for a rolling-midnight console
 * launch? The case file tags each comparable with a tier in its unlock
 * note; only the clean "comparable" tier is kept, launches whose unlock was
 * one simultaneous global moment are dropped (their first hour is before
 * the unlock, so their shape is a different animal), and the target game's
 * own pre-release history is dropped (it is not a launch at all).
 */
function isUsableAnalogue(c: Comparable): boolean {
  if (c.name.includes("Grand Theft Auto VI")) return false;
  const note = (c.unlock ?? "").toLowerCase();
  const tier = /tier:\s*([a-z-]+)/.exec(note);
  if (!tier || tier[1] !== "comparable") return false;
  if (note.includes("imultaneous") || note.includes("single global") || note.includes("single time")) return false;
  return true;
}

/**
 * A comparable's hourly points re-indexed by hour of its release day
 * (1..24), from its day_start. Points outside the release day are ignored.
 */
function releaseDayHours(c: Comparable): Map<number, number> {
  const out = new Map<number, number>();
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return out;
  for (const p of c.hourly ?? []) {
    if (typeof p.viewers !== "number" || !(p.viewers > 0)) continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !out.has(h)) out.set(h, p.viewers);
  }
  return out;
}

/** The house persistence baseline for one hour, used only as a fallback. */
function baselineRow(p1: number, h: number): number[] {
  return Z_SCORES.map((z) => round2(p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1))));
}

/**
 * The five quantiles for one hour, blended from the ratio median and the
 * level regression over the usable analogues. Returns null when the
 * analogues carry no information for this hour (caller falls back).
 */
function forecastRow(
  analogues: Array<Map<number, number>>,
  logP1: number,
  h: number,
): number[] | null {
  const logRatios: number[] = [];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const day of analogues) {
    const v1 = day.get(1);
    const vh = day.get(h);
    if (v1 === undefined || vh === undefined || v1 <= 0 || vh <= 0) continue;
    logRatios.push(Math.log(vh / v1));
    xs.push(Math.log(v1));
    ys.push(Math.log(vh));
  }
  if (logRatios.length === 0) return null;
  logRatios.sort((a, b) => a - b);
  const medianRatio = quantileOf(logRatios, 0.5);
  // With only one or two analogues the interquartile spread is meaningless,
  // so widen like the baseline instead of trusting it.
  const spread =
    logRatios.length >= 4
      ? Math.max((quantileOf(logRatios, 0.75) - quantileOf(logRatios, 0.25)) / 1.349, 0.05) * SPREAD_KEEP
      : BASELINE_SIGMA * Math.sqrt(h - 1);

  // Median from persistence: hour 1 scaled by the typical ratio.
  const medianRatioOnly = logP1 + medianRatio;
  // Median from the level regression (only when enough points fit it).
  let median = medianRatioOnly;
  if (xs.length >= 5) {
    const [a, b] = fitLine(xs, ys);
    const medianReg = a + b * logP1;
    if (Number.isFinite(medianReg)) {
      median = REGRESSION_WEIGHT * medianReg + (1 - REGRESSION_WEIGHT) * medianRatioOnly;
    }
  }
  if (!Number.isFinite(median) || !Number.isFinite(spread)) return null;

  const row = Z_SCORES.map((z) => round2(Math.exp(median + z * spread)));
  // Clamp to the contest's bounds and enforce non-decreasing quantiles.
  let prev = 1;
  for (let i = 0; i < row.length; i++) {
    let v = row[i];
    if (!Number.isFinite(v) || v < 1) v = 1;
    if (v > 1e9) v = 1e9;
    if (v < prev) v = prev;
    row[i] = v;
    prev = v;
  }
  return row;
}

function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, first !== undefined && first.viewers > 0 ? first.viewers : 1);
  const logP1 = Math.log(p1);
  const analogues = (c.comparables ?? []).filter(isUsableAnalogue).map(releaseDayHours);
  const hours = (c.forecast ?? []).map(({ hour }) => {
    const row = forecastRow(analogues, logP1, hour) ?? baselineRow(p1, hour);
    return { hour, q: row };
  });
  // The forecast list must hold hours 2..24 exactly once; repair it if the
  // case file ever lists them oddly.
  const seen = new Set(hours.map((r) => r.hour));
  for (let h = 2; h <= 24; h++) {
    if (!seen.has(h)) hours.push({ hour: h, q: baselineRow(p1, h) });
  }
  hours.sort((a, b) => a.hour - b.hour);
  return { hours };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  let out: { hours: Array<{ hour: number; q: number[] }> };
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
    out = forecastCase(c);
  } catch {
    // Never leave a case without a valid forecast: fall back to the
    // persistence baseline around 1 viewer.
    out = { hours: Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, q: baselineRow(1, i + 2) })) };
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
