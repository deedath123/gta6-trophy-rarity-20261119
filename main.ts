// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// at the case's target as five quantiles, scored by pinball loss on the logit scale.
//
// Model (counts, not rates): a trophy's rate is E_i(t) / N(t), earned counts over
// players tracked. Exophase refreshes its page every few hours, so between refreshes
// reads are stale and at a refresh N jumps and rates fall; fitting E and N separately
// handles that dilution instead of chasing it as a rate trend.
//
// Candidates, picked by a rolling-origin CV that predicts the LAST read of the
// fitting window (up to ~18h ahead -- a short-horizon gate would pick persistence
// even when a trend pays over the 4.5-day final horizon):
//   A: persistence (median = last rate).
//   B: counts log-linear: ln N and ln E_i linear in ln(1+h), per-trophy slopes
//      shrunk toward tier means, anchored at the last read.
//   C: damped B (slopes x0.6, median capped at tier ceilings).
//   D: saturating counts: E_i(t) = N(t) * R_i * (1 - exp(-h/tau_i)), tau grid-searched.
//   E: logit-blend of D and persistence (50/50).
//   F: damped B (slopes x0.5).
// The fit window is the last 24h before the checkpoint: fitting the full history
// with a log-linear form extrapolates launch-night explosive growth and overshoots.
// Spread = slope-SE x horizon + refresh-jump noise + random-walk floor, so a
// forecast that is too narrow pays for every trophy outside it and one too wide
// pays at every trophy.
// Flat-history guard: when N never changed in the whole history, a refresh is due
// before the target and the new cohort dilutes rates -- medians shift x0.2 with a
// wide spread (the first observed refresh multiplied N by ~6, so rates fall to
// ~1/6; the practice cases are exactly this regime). A trophy at 100% stays ~100%.
// Node 22, standard library only; deterministic (no clock, no randomness).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const CEIL: Record<string, number> = { platinum: 5, gold: 40, silver: 70, bronze: 95 };
const T0_MS = 1795027200000; // 2026-11-19T00:00:00Z

interface Trophy { id: number; tier: string; }
interface Read { at: string; players_tracked: number; rates: Array<number | null>; }
interface Case {
  id: string; at: string; target: string; horizon_minutes: number;
  levels: number[]; trophies: Trophy[]; history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

function hours(at: string): number {
  return (Date.parse(at) - T0_MS) / 3600000;
}

function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, sxy = 0;
  for (let k = 0; k < n; k++) { sxx += (xs[k] - mx) ** 2; sxy += (xs[k] - mx) * (ys[k] - my); }
  if (sxx <= 1e-12) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

function pinball(y: number, q: number, tau: number): number {
  const d = logit(y) - logit(q);
  return 2 * Math.max(tau * d, (tau - 1) * d);
}

function medianA(c: Case): Record<number, number> {
  const last = c.history[c.history.length - 1];
  const out: Record<number, number> = {};
  c.trophies.forEach((t, i) => { out[t.id] = last.rates[i] ?? 0; });
  return out;
}

function quantiles(median: number, spread: number, rLast: number | null): number[] {
  // A trophy the last read shows at ~100% stays ~100% only when the model agrees
  // (median >= 99): if the counts model itself forecasts a drop (dilution by new
  // players who have not earned it yet), trust the model instead of forcing 100.
  if (rLast !== null && rLast >= 99.99 && median >= 99) return [99.5, 99.9, 100, 100, 100];
  const l = logit(Math.min(Math.max(median, 0.01), 99.99));
  return Z.map((z) => round4(invLogit(l + z * spread)));
}

interface FitCounts { med: Record<number, number>; resid: number[]; }
function fitCounts(c: Case, idxs: number[], damp: number): FitCounts {
  const hist = c.history;
  const xs = idxs.map((j) => Math.log1p(Math.max(hours(hist[j].at), 0)));
  const xLast = xs[xs.length - 1];
  const xT = Math.log1p(Math.max(hours(c.target), 0));
  const Ns = idxs.map((j) => hist[j].players_tracked);
  const [aN, bN] = ols(xs, Ns.map((n) => Math.log(n)));
  const NT = Math.exp(aN + bN * xT);
  const slopes: number[] = [];
  const intercepts: number[] = [];
  const resid: number[] = [];
  c.trophies.forEach((t, i) => {
    const fx: number[] = [];
    const fy: number[] = [];
    idxs.forEach((j, k) => {
      const rate = hist[j].rates[i];
      if (rate === null) return;
      const E = (rate * hist[j].players_tracked) / 100;
      if (E > 0.5) { fx.push(xs[k]); fy.push(Math.log(E)); }
    });
    let a = 0, b = 0;
    if (fx.length >= 2 && Math.max(...fx) > Math.min(...fx)) [a, b] = ols(fx, fy);
    b *= damp;
    const eLast = ((hist[idxs[idxs.length - 1]].rates[i] ?? 0) * Ns[Ns.length - 1]) / 100;
    a = Math.log(Math.max(eLast, 0.5)) - b * xLast;
    slopes.push(b);
    intercepts.push(a);
    // slope standard error
    const res: number[] = [];
    const xsR: number[] = [];
    idxs.forEach((j, k) => {
      const rate = hist[j].rates[i];
      if (rate === null) return;
      const E = (rate * hist[j].players_tracked) / 100;
      res.push(Math.log(Math.max(E, 0.5)) - (a + b * xs[k]));
      xsR.push(xs[k]);
    });
    let se = 0;
    if (res.length >= 3) {
      const m = res.reduce((x, y) => x + y, 0) / res.length;
      const sd = Math.sqrt(res.reduce((s, v) => s + (v - m) ** 2, 0) / (res.length - 2));
      const mx = xsR.reduce((x, y) => x + y, 0) / xsR.length;
      const sxx = xsR.reduce((s, x) => s + (x - mx) ** 2, 0);
      se = sxx > 1e-12 ? sd / Math.sqrt(sxx) : 0;
    }
    resid.push(se);
  });
  const tierSlopes: Record<string, number[]> = {};
  c.trophies.forEach((t, i) => { (tierSlopes[t.tier] ??= []).push(slopes[i]); });
  const tierMean: Record<string, number> = {};
  for (const k of Object.keys(tierSlopes)) {
    tierMean[k] = tierSlopes[k].reduce((a, b) => a + b, 0) / tierSlopes[k].length;
  }
  const med: Record<number, number> = {};
  c.trophies.forEach((t, i) => {
    const w = idxs.length / (idxs.length + 8);
    const b = w * slopes[i] + (1 - w) * tierMean[t.tier];
    const a = intercepts[i] + (b - slopes[i]) * xLast;
    med[t.id] = (100 * Math.exp(a + b * xT)) / NT;
  });
  return { med, resid };
}

interface LinFit { aN: number; bN: number; ab: Array<[number, number]>; }
function fitFor(c: Case, fitIdxs: number[], damp: number): LinFit {
  const hist = c.history;
  const xs = fitIdxs.map((j) => Math.log1p(Math.max(hours(hist[j].at), 0)));
  const xLast = xs[xs.length - 1];
  const Ns = fitIdxs.map((j) => hist[j].players_tracked);
  const [aN, bN] = ols(xs, Ns.map((n) => Math.log(n)));
  const ab: Array<[number, number]> = [];
  c.trophies.forEach((t, i) => {
    const fx: number[] = [];
    const fy: number[] = [];
    fitIdxs.forEach((j, k) => {
      const rate = hist[j].rates[i];
      if (rate === null) return;
      const E = (rate * hist[j].players_tracked) / 100;
      if (E > 0.5) { fx.push(xs[k]); fy.push(Math.log(E)); }
    });
    let b = 0;
    if (fx.length >= 2 && Math.max(...fx) > Math.min(...fx)) [, b] = ols(fx, fy);
    b *= damp;
    const eLast = ((hist[fitIdxs[fitIdxs.length - 1]].rates[i] ?? 0) * Ns[Ns.length - 1]) / 100;
    const a = Math.log(Math.max(eLast, 0.5)) - b * xLast;
    ab.push([a, b]);
  });
  const tierSlopes: Record<string, number[]> = {};
  c.trophies.forEach((t, i) => { (tierSlopes[t.tier] ??= []).push(ab[i][1]); });
  const tierMean: Record<string, number> = {};
  for (const k of Object.keys(tierSlopes)) {
    tierMean[k] = tierSlopes[k].reduce((a, b) => a + b, 0) / tierSlopes[k].length;
  }
  c.trophies.forEach((t, i) => {
    const w = fitIdxs.length / (fitIdxs.length + 8);
    const bNew = w * ab[i][1] + (1 - w) * tierMean[t.tier];
    ab[i][0] += (bNew - ab[i][1]) * xLast;
    ab[i][1] = bNew;
  });
  return { aN, bN, ab };
}

function medianAt(c: Case, fit: LinFit, x: number): Record<number, number> {
  const Np = Math.exp(fit.aN + fit.bN * x);
  const out: Record<number, number> = {};
  c.trophies.forEach((t, i) => {
    const [a, b] = fit.ab[i];
    out[t.id] = (100 * Math.exp(a + b * x)) / Np;
  });
  return out;
}

interface SatFit {
  pred: (xLog1p: number, hTarget: number) => Record<number, number>;
  resid: number[];
}
function fitSat(c: Case, fitIdxs: number[]): SatFit {
  const hist = c.history;
  const xs = fitIdxs.map((j) => Math.log1p(Math.max(hours(hist[j].at), 0)));
  const hs = fitIdxs.map((j) => Math.max(hours(hist[j].at), 0));
  const Ns = fitIdxs.map((j) => hist[j].players_tracked);
  const [aN, bN] = ols(xs, Ns.map((n) => Math.log(n)));
  const taus = [6, 12, 24, 48, 96, 192, 384, 768];
  const Rs: number[] = [];
  const TAUs: number[] = [];
  const resid: number[] = [];
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sxx = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  c.trophies.forEach((t, i) => {
    let best: [number, number, number, number] | null = null;
    for (const tau of taus) {
      const S = hs.map((h) => 1 - Math.exp(-h / tau));
      let num = 0, den = 0;
      fitIdxs.forEach((j, k) => {
        const rate = hist[j].rates[i];
        if (rate === null) return;
        const E = (rate * hist[j].players_tracked) / 100;
        num += E * hist[j].players_tracked * S[k];
        den += (hist[j].players_tracked * S[k]) ** 2;
      });
      if (den <= 0) continue;
      const Rr = num / den;
      let sse = 0, cnt = 0;
      fitIdxs.forEach((j, k) => {
        const rate = hist[j].rates[i];
        if (rate === null) return;
        const E = (rate * hist[j].players_tracked) / 100;
        const predE = hist[j].players_tracked * Rr * S[k];
        sse += (Math.log(Math.max(E, 0.5)) - Math.log(Math.max(predE, 0.5))) ** 2;
        cnt++;
      });
      if (best === null || sse < best[0]) best = [sse, Rr, tau, cnt];
    }
    if (best === null) {
      Rs.push(0); TAUs.push(48); resid.push(0);
    } else {
      const [sse, Rr, tau, cnt] = best;
      Rs.push(Rr); TAUs.push(tau);
      const sd = Math.sqrt(sse / Math.max(cnt - 2, 1));
      resid.push(sxx > 1e-12 ? sd / Math.sqrt(sxx) : 0);
    }
  });
  const tierR: Record<string, number[]> = {};
  const tierTau: Record<string, number[]> = {};
  c.trophies.forEach((t, i) => {
    (tierR[t.tier] ??= []).push(Rs[i]);
    (tierTau[t.tier] ??= []).push(Math.log(TAUs[i]));
  });
  const tierRMed: Record<string, number> = {};
  const tierTauMed: Record<string, number> = {};
  for (const k of Object.keys(tierR)) {
    const s = [...tierR[k]].sort((a, b) => a - b);
    tierRMed[k] = s[Math.floor(s.length / 2)];
    const s2 = [...tierTau[k]].sort((a, b) => a - b);
    tierTauMed[k] = Math.exp(s2[Math.floor(s2.length / 2)]);
  }
  c.trophies.forEach((t, i) => {
    Rs[i] = 0.8 * Rs[i] + 0.2 * tierRMed[t.tier];
    TAUs[i] = Math.exp(0.8 * Math.log(TAUs[i]) + 0.2 * Math.log(tierTauMed[t.tier]));
  });
  const pred = (xLog1p: number, hTarget: number): Record<number, number> => {
    const out: Record<number, number> = {};
    c.trophies.forEach((t, i) => {
      out[t.id] = 100 * Rs[i] * (1 - Math.exp(-Math.max(hTarget, 0) / TAUs[i]));
    });
    return out;
  };
  return { pred, resid };
}

function spreadFromResid(c: Case, resid: number[], days: number): Record<number, number> {
  const hist = c.history;
  const jumps: number[] = [];
  for (let j = 1; j < hist.length; j++) {
    if (hist[j].players_tracked === hist[j - 1].players_tracked) continue;
    c.trophies.forEach((t, i) => {
      const r0 = hist[j - 1].rates[i];
      const r1 = hist[j].rates[i];
      if (r0 === null || r1 === null) return;
      jumps.push(Math.abs(logit(r1) - logit(r0)));
    });
  }
  jumps.sort((a, b) => a - b);
  const sJump = jumps.length ? jumps[Math.floor(jumps.length / 2)] : 0;
  const tierResid: Record<string, number[]> = {};
  c.trophies.forEach((t, i) => { (tierResid[t.tier] ??= []).push(resid[i]); });
  const tierMed: Record<string, number> = {};
  for (const k of Object.keys(tierResid)) {
    const s = [...tierResid[k]].sort((a, b) => a - b);
    tierMed[k] = s[Math.floor(s.length / 2)];
  }
  const out: Record<number, number> = {};
  c.trophies.forEach((t, i) => {
    const se = 0.5 * resid[i] + 0.5 * tierMed[t.tier];
    const s = Math.sqrt((se * 3) ** 2 + sJump * sJump + (0.5 * Math.sqrt(days)) ** 2);
    out[t.id] = Math.min(Math.max(s, 0.3), 1.5);
  });
  return out;
}

function forecast(c: Case): Record<number, number[]> {
  const hist = c.history;
  const days = c.horizon_minutes / 1440;
  const NsAll = hist.map((r) => r.players_tracked);
  const flatN = Math.max(...NsAll) === Math.min(...NsAll);
  const tCp = hours(c.at);
  let win = hist.map((_, j) => j).filter((j) => tCp - hours(hist[j].at) <= 24 + 1e-9);
  if (win.length < 3) win = hist.map((_, j) => j);
  const rw = 0.5 * Math.sqrt(days);

  if (flatN || win.length < 6) {
    const med = medianA(c);
    const out: Record<number, number[]> = {};
    c.trophies.forEach((t, i) => {
      const rLast = hist[hist.length - 1].rates[i];
      if (flatN && rLast !== null && rLast < 99.99) {
        out[t.id] = quantiles(med[t.id] * 0.2, Math.max(rw, 0.6), rLast);
      } else {
        out[t.id] = quantiles(med[t.id], Math.max(rw, 0.25), rLast);
      }
    });
    return out;
  }

  const cvLoss = (kind: string, damp = 1): number => {
    let total = 0, n = 0;
    const L = win.length;
    for (const frac of [0.25, 0.4]) {
      const k = Math.max(3, Math.floor(L * frac));
      if (k >= L - 1) continue;
      const fitIdxs = win.slice(0, k);
      const evalIdx = win[win.length - 1];
      let pred: (x: number) => Record<number, number>;
      let resid: number[];
      if (kind === 'A') {
        const med0 = medianA({ ...c, history: fitIdxs.map((j) => hist[j]) });
        resid = c.trophies.map(() => 0);
        pred = () => ({ ...med0 });
      } else if (kind === 'D' || kind === 'E') {
        const sat = fitSat(c, fitIdxs);
        resid = sat.resid;
        if (kind === 'D') {
          pred = (x) => sat.pred(x, Math.expm1(x));
        } else {
          const med0 = medianA({ ...c, history: fitIdxs.map((j) => hist[j]) });
          pred = (x) => {
            const dm = sat.pred(x, Math.expm1(x));
            const out: Record<number, number> = {};
            c.trophies.forEach((t) => {
              out[t.id] = invLogit(0.5 * logit(dm[t.id]) + 0.5 * logit(med0[t.id]));
            });
            return out;
          };
        }
      } else {
        const fit = fitFor(c, fitIdxs, damp);
        const fc = fitCounts({ ...c, target: hist[evalIdx].at, horizon_minutes: 10 }, fitIdxs, damp);
        resid = fc.resid;
        pred = (x) => medianAt(c, fit, x);
      }
      const spr0 = spreadFromResid(c, resid, 10 / 1440);
      const x = Math.log1p(Math.max(hours(hist[evalIdx].at), 0));
      const hDays = Math.max((hours(hist[evalIdx].at) - hours(hist[fitIdxs[fitIdxs.length - 1]].at)) / 24, 0);
      const med = pred(x);
      c.trophies.forEach((t, i) => {
        const y = hist[evalIdx].rates[i];
        if (y === null) return;
        let s = Math.sqrt(spr0[t.id] ** 2 + (0.5 * Math.sqrt(hDays)) ** 2);
        s = Math.min(Math.max(s, 0.25), 2);
        let m = med[t.id];
        if (kind === 'C') m = Math.min(m, CEIL[t.tier]);
        const q = quantiles(m, s, hist[fitIdxs[fitIdxs.length - 1]].rates[i]);
        for (let li = 0; li < 5; li++) { total += pinball(y, q[li], LEVELS[li]); n++; }
      });
    }
    return total / Math.max(n, 1);
  };

  const cands: Array<[string, number]> = [
    ['A', cvLoss('A')],
    ['B', cvLoss('B', 1)],
    ['C', cvLoss('C', 0.6)],
    ['D', cvLoss('D')],
    ['E', cvLoss('E')],
    ['F', cvLoss('F', 0.5)],
  ];
  cands.sort((a, b) => a[1] - b[1]);
  const best = cands[0][0];
  let med: Record<number, number>;
  let resid: number[];
  if (best === 'A') {
    med = medianA(c);
    resid = c.trophies.map(() => 0);
  } else if (best === 'D' || best === 'E') {
    const sat = fitSat(c, win);
    resid = sat.resid;
    const xT = Math.log1p(Math.max(hours(c.target), 0));
    med = sat.pred(xT, Math.max(hours(c.target), 0));
    c.trophies.forEach((t) => { med[t.id] = Math.min(med[t.id], CEIL[t.tier]); });
    if (best === 'E') {
      const med0 = medianA(c);
      c.trophies.forEach((t) => {
        med[t.id] = invLogit(0.5 * logit(med[t.id]) + 0.5 * logit(med0[t.id]));
      });
    }
  } else if (best === 'B') {
    ({ med, resid } = fitCounts(c, win, 1));
  } else if (best === 'F') {
    ({ med, resid } = fitCounts(c, win, 0.5));
  } else {
    ({ med, resid } = fitCounts(c, win, 0.6));
    c.trophies.forEach((t) => { med[t.id] = Math.min(med[t.id], CEIL[t.tier]); });
  }
  const spr = spreadFromResid(c, resid, days);
  const out: Record<number, number[]> = {};
  c.trophies.forEach((t, i) => {
    out[t.id] = quantiles(med[t.id], spr[t.id], hist[hist.length - 1].rates[i]);
  });
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  const trophies = c.trophies.map((t) => ({ id: t.id, q: forecast(c)[t.id] }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
