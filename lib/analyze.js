'use strict';
/*
 * Shared analytics: turn the collected top-performer positions into
 *  - a "common positions" signal (what smart money agrees on), and
 *  - a budget allocation (a spread of trades given a bankroll + risk profile).
 *
 * Pure functions, no I/O, no dependencies. Used by both collect.js (CLI) and
 * server.js (dashboard).
 */

// Aggregate identical market+outcome holdings across the pool of top traders.
// Returns rows sorted by conviction (holder count) then aggregate value.
function aggregateCommon(users) {
  const map = new Map();
  for (const u of users) {
    for (const p of u.positions || []) {
      const key = `${p.conditionId}::${p.outcome}`;
      if (!map.has(key)) {
        map.set(key, {
          key,
          conditionId: p.conditionId,
          title: p.title,
          outcome: p.outcome,
          slug: p.slug,
          eventSlug: p.eventSlug,
          endDate: p.endDate,
          curPrice: p.curPrice,
          holders: [],
          totalShares: 0,
          totalValue: 0,
          avgEntrySum: 0,
        });
      }
      const e = map.get(key);
      e.holders.push({
        name: u.name,
        wallet: u.wallet,
        size: p.size,
        avgPrice: p.avgPrice,
        value: p.currentValue,
        pnl: p.cashPnl,
        pctPnl: p.percentPnl,
      });
      e.totalShares += p.size;
      e.totalValue += p.currentValue;
      e.avgEntrySum += p.avgPrice;
      e.curPrice = p.curPrice; // latest observed
    }
  }
  return [...map.values()]
    .map((e) => ({
      ...e,
      holderCount: e.holders.length,
      avgEntry: e.avgEntrySum / e.holders.length,
    }))
    .sort((a, b) => b.holderCount - a.holderCount || b.totalValue - a.totalValue);
}

// --- scoring helpers ---
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const normList = (arr, sel) => {
  const vals = arr.map(sel);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  return (i) => (sel(arr[i]) - min) / span;
};

// Risk profiles weight the four signals differently.
//   conviction  = how many top traders hold it
//   capital     = aggregate smart-money $ in it (log-scaled)
//   room        = price headroom, peaks at 0.5 (not-yet-decided markets)
//   momentum    = curPrice - avgEntry (smart money already in profit)
const RISK_WEIGHTS = {
  conservative: { conviction: 0.4, capital: 0.3, room: 0.05, momentum: 0.25, likelihood: 0.35 },
  balanced: { conviction: 0.35, capital: 0.25, room: 0.2, momentum: 0.2, likelihood: 0.1 },
  aggressive: { conviction: 0.25, capital: 0.15, room: 0.4, momentum: 0.2, likelihood: -0.1 },
};

/*
 * Score candidate markets. Only considers positions that are still tradeable
 * (price strictly between the guard rails) and held by >= minHolders traders.
 */
function scoreCandidates(common, { risk = 'balanced', minHolders = 2, minPrice = 0.05, maxPrice = 0.95 } = {}) {
  const w = RISK_WEIGHTS[risk] || RISK_WEIGHTS.balanced;
  const pool = common.filter(
    (c) => c.holderCount >= minHolders && c.curPrice > minPrice && c.curPrice < maxPrice
  );
  if (!pool.length) return [];

  const nConv = normList(pool, (c) => c.holderCount);
  const nCap = normList(pool, (c) => Math.log10(c.totalValue + 10));
  const nMom = normList(pool, (c) => c.curPrice - c.avgEntry);

  return pool
    .map((c, i) => {
      const room = 4 * c.curPrice * (1 - c.curPrice); // 0..1, peaks at p=0.5
      const likelihood = c.curPrice; // higher price = more likely per market
      const score =
        w.conviction * nConv(i) +
        w.capital * nCap(i) +
        w.room * room +
        w.momentum * clamp01((nMom(i) + 1) / 2) +
        w.likelihood * likelihood;
      return {
        ...c,
        signals: {
          conviction: +nConv(i).toFixed(3),
          capital: +nCap(i).toFixed(3),
          room: +room.toFixed(3),
          momentum: +(c.curPrice - c.avgEntry).toFixed(3),
          likelihood: +likelihood.toFixed(3),
        },
        score: +score.toFixed(4),
      };
    })
    .sort((a, b) => b.score - a.score);
}

/*
 * Allocate a bankroll across the top-scoring candidates.
 * - diversifies (at most one pick per event, configurable),
 * - caps any single stake at maxPerTradePct of the budget,
 * - stakes proportional to score, rounded to cents, min $1.
 */
function allocate(common, { budget = 100, risk = 'balanced', maxPicks = 10, maxPerTradePct = 0.2, oneRowPerEvent = true } = {}) {
  let scored = scoreCandidates(common, { risk });

  if (oneRowPerEvent) {
    const seen = new Set();
    scored = scored.filter((c) => {
      const ev = c.eventSlug || c.conditionId;
      if (seen.has(ev)) return false;
      seen.add(ev);
      return true;
    });
  }

  const picks = scored.slice(0, maxPicks);
  if (!picks.length) return { budget, risk, picks: [], deployed: 0, reserved: budget, note: 'No tradeable candidates matched the filters.' };

  const scoreSum = picks.reduce((s, p) => s + p.score, 0) || 1;
  const cap = budget * maxPerTradePct;

  // Proportional stakes with per-trade cap, then redistribute leftover once.
  let stakes = picks.map((p) => Math.min(cap, (p.score / scoreSum) * budget));
  let deployed = stakes.reduce((a, b) => a + b, 0);
  const leftover = budget - deployed;
  if (leftover > 0.01) {
    // spread leftover to uncapped picks proportionally
    const room = picks.map((p, i) => cap - stakes[i]);
    const roomSum = room.reduce((a, b) => a + b, 0) || 1;
    stakes = stakes.map((s, i) => s + leftover * (room[i] / roomSum));
  }

  const out = picks.map((p, i) => {
    const stake = Math.max(1, Math.round(stakes[i] * 100) / 100);
    const shares = stake / p.curPrice;
    const payout = shares * 1.0; // each share resolves to $1 if correct
    const profit = payout - stake;
    const roiPct = (1 / p.curPrice - 1) * 100;
    return {
      title: p.title,
      outcome: p.outcome,
      side: `${p.outcome} @ ${(p.curPrice * 100).toFixed(1)}¢`,
      conditionId: p.conditionId,
      slug: p.slug,
      eventSlug: p.eventSlug,
      endDate: p.endDate,
      curPrice: p.curPrice,
      avgEntry: p.avgEntry,
      holderCount: p.holderCount,
      smartMoney: Math.round(p.totalValue),
      score: p.score,
      signals: p.signals,
      stake: +stake.toFixed(2),
      allocationPct: +((stake / budget) * 100).toFixed(1),
      shares: +shares.toFixed(1),
      potentialPayout: +payout.toFixed(2),
      potentialProfit: +profit.toFixed(2),
      roiIfWin: +roiPct.toFixed(1),
      rationale: buildRationale(p),
    };
  });

  // Correct any rounding overflow so we never deploy more than the budget.
  // Use integer-cent math to avoid floating-point sum artifacts.
  const centsSum = (arr) => arr.reduce((s, p) => s + Math.round(p.stake * 100), 0);
  const overflowCents = centsSum(out) - Math.round(budget * 100);
  if (overflowCents > 0) {
    const big = out.reduce((a, b) => (b.stake > a.stake ? b : a), out[0]);
    big.stake = +(big.stake - overflowCents / 100).toFixed(2);
    big.allocationPct = +((big.stake / budget) * 100).toFixed(1);
    big.shares = +(big.stake / big.curPrice).toFixed(1);
    big.potentialPayout = +(big.shares * 1.0).toFixed(2);
    big.potentialProfit = +(big.potentialPayout - big.stake).toFixed(2);
  }
  const totalDeployed = centsSum(out) / 100;
  return {
    budget,
    risk,
    generatedAt: null, // stamped by caller
    picks: out,
    deployed: +totalDeployed.toFixed(2),
    reserved: Math.max(0, +(budget - totalDeployed).toFixed(2)),
    candidateCount: scored.length,
  };
}

function buildRationale(p) {
  const bits = [];
  bits.push(`${p.holderCount} of the top traders hold this`);
  bits.push(`$${Math.round(p.totalValue).toLocaleString()} smart-money committed`);
  const edge = p.curPrice - p.avgEntry;
  if (edge > 0.02) bits.push(`already up ${(edge * 100).toFixed(0)}¢ from their avg entry (momentum)`);
  else if (edge < -0.02) bits.push(`below their avg entry — potential dip-buy at ${(p.curPrice * 100).toFixed(0)}¢`);
  const room = 4 * p.curPrice * (1 - p.curPrice);
  if (room > 0.7) bits.push(`priced near a coin-flip (${(p.curPrice * 100).toFixed(0)}¢) — lots of headroom`);
  else if (p.curPrice > 0.8) bits.push(`high-confidence favorite (${(p.curPrice * 100).toFixed(0)}¢)`);
  return bits.join('; ') + '.';
}

module.exports = { aggregateCommon, scoreCandidates, allocate, buildRationale, RISK_WEIGHTS };
