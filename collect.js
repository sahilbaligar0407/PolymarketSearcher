#!/usr/bin/env node
/*
 * Polymarket Top-Performers Position Collector
 * --------------------------------------------
 * Pure API, zero dependencies (needs Node 18+ for global fetch; tested on Node 24).
 *
 * Builds a pool of top-performing wallets from the public leaderboard boards
 * (the public API hard-caps each board at 50, so we UNION several boards to reach
 * ~100 unique top performers), then pulls each wallet's meaningful OPEN positions
 * from the public positions API and writes three outputs:
 *
 *   output/top100_positions_<date>.md    human-readable mega document
 *   output/top100_positions_<date>.json  structured data (for ChatGPT upload)
 *   output/common_positions_<date>.md    cross-user commonality table (basket-copy signal)
 *
 * Run:  node collect.js      (or)  npm start
 */

'use strict';
const fs = require('fs');
const path = require('path');

// ------------------------------- CONFIG --------------------------------------
const CONFIG = {
  TARGET_USERS: 100,          // size of the top-performer pool
  WINDOW: '30d',              // leaderboard window: 1d | 7d | 30d | all
  // Boards unioned to build the pool, in priority order. First board defines the
  // primary rank; later boards fill remaining slots. metric: profit | volume.
  BOARDS: [
    { metric: 'profit', window: '30d', label: 'Monthly Profit' },
    { metric: 'volume', window: '30d', label: 'Monthly Volume' },
    { metric: 'profit', window: 'all', label: 'All-time Profit' },
    { metric: 'volume', window: 'all', label: 'All-time Volume' },
  ],
  MIN_POSITION_USD: 50,       // ignore open positions worth less than this
  MAX_POSITIONS_PER_USER: 40, // keep at most this many (largest by current value)
  CONCURRENCY: 6,             // parallel position requests
  REQUEST_DELAY_MS: 60,       // small delay between requests (rate-limit courtesy)
  OUTPUT_DIR: path.join(__dirname, 'output'),
};

const LB_API = 'https://lb-api.polymarket.com';
const DATA_API = 'https://data-api.polymarket.com';

// ------------------------------- HELPERS -------------------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function today() {
  // Local date as YYYY-MM-DD
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

async function getJSON(url, { retries = 3 } = {}) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': 'PolymarketSearcher/1.0' } });
      if (res.status === 429) {
        const wait = 1000 * (attempt + 1);
        process.stderr.write(`  rate-limited, waiting ${wait}ms...\n`);
        await sleep(wait);
        continue;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return await res.json();
    } catch (err) {
      if (attempt === retries) throw err;
      await sleep(500 * (attempt + 1));
    }
  }
}

const usd = (n) =>
  (n < 0 ? '-$' : '$') +
  Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const cents = (p) => `${(p * 100).toFixed(1)}¢`;
const shares = (n) => n.toLocaleString('en-US', { maximumFractionDigits: 1 });
const pct = (n) => `${n >= 0 ? '' : ''}${n.toFixed(2)}%`;

// Run async tasks with a fixed concurrency limit; reports progress.
async function pool(items, concurrency, worker, onProgress) {
  const results = new Array(items.length);
  let index = 0;
  let done = 0;
  async function runner() {
    while (index < items.length) {
      const i = index++;
      try {
        results[i] = await worker(items[i], i);
      } catch (err) {
        results[i] = { error: err.message };
      }
      done++;
      if (onProgress) onProgress(done, items.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, runner));
  return results;
}

// ------------------------- STEP 1: BUILD USER POOL ---------------------------
async function buildUserPool() {
  console.log(`\n[1/3] Building top-${CONFIG.TARGET_USERS} performer pool from leaderboard boards...`);
  const byWallet = new Map(); // wallet -> user record

  for (const board of CONFIG.BOARDS) {
    const url = `${LB_API}/${board.metric}?window=${board.window}&limit=100`;
    let rows;
    try {
      rows = await getJSON(url);
    } catch (err) {
      console.warn(`  ! ${board.label}: ${err.message}`);
      continue;
    }
    console.log(`  ${board.label.padEnd(16)} -> ${rows.length} rows`);
    rows.forEach((row, i) => {
      const wallet = row.proxyWallet.toLowerCase();
      const rank = i + 1;
      const name = row.name || row.pseudonym || wallet;
      if (!byWallet.has(wallet)) {
        byWallet.set(wallet, {
          wallet,
          name,
          pseudonym: row.pseudonym || null,
          boards: {},
          firstBoard: board.label,
          firstRank: rank,
        });
      }
      byWallet.get(wallet).boards[board.label] = { rank, amount: row.amount };
    });
  }

  // Order: users appearing on earlier (higher-priority) boards first, then by rank.
  const boardPriority = new Map(CONFIG.BOARDS.map((b, i) => [b.label, i]));
  const ordered = [...byWallet.values()].sort((a, b) => {
    const pa = boardPriority.get(a.firstBoard);
    const pb = boardPriority.get(b.firstBoard);
    if (pa !== pb) return pa - pb;
    return a.firstRank - b.firstRank;
  });

  const pooled = ordered.slice(0, CONFIG.TARGET_USERS);
  console.log(`  => ${byWallet.size} unique performers found; taking top ${pooled.length}.`);
  return pooled;
}

// ------------------- STEP 2: FETCH POSITIONS PER USER ------------------------
function isOpen(p) {
  return (
    !p.redeemable &&
    p.curPrice > 0 &&
    p.curPrice < 1 &&
    p.size > 0.01 &&
    p.currentValue >= CONFIG.MIN_POSITION_USD
  );
}

async function fetchUserPositions(user) {
  const url =
    `${DATA_API}/positions?user=${user.wallet}` +
    `&sortBy=CURRENT&sortDirection=DESC&limit=500`;
  await sleep(CONFIG.REQUEST_DELAY_MS);
  const raw = await getJSON(url);
  const open = raw
    .filter(isOpen)
    .slice(0, CONFIG.MAX_POSITIONS_PER_USER)
    .map((p) => ({
      title: p.title,
      slug: p.slug,
      eventSlug: p.eventSlug,
      outcome: p.outcome,
      curPrice: p.curPrice,
      avgPrice: p.avgPrice,
      size: p.size,
      currentValue: p.currentValue,
      cashPnl: p.cashPnl,
      percentPnl: p.percentPnl,
      endDate: p.endDate,
      conditionId: p.conditionId,
      asset: p.asset,
    }));
  const totalValue = open.reduce((s, p) => s + p.currentValue, 0);
  const totalPnl = open.reduce((s, p) => s + p.cashPnl, 0);
  return { ...user, positions: open, totalValue, totalPnl, rawCount: raw.length };
}

// --------------------------- STEP 3: RENDER ----------------------------------
function renderMarkdown(users, date) {
  const lines = [];
  lines.push(`# Polymarket Top Performers — Open Positions`);
  lines.push('');
  lines.push(`Generated: ${date}`);
  lines.push(`Leaderboard pool: ${CONFIG.BOARDS.map((b) => b.label).join(' ∪ ')}`);
  lines.push(
    `Filters: open positions only, current value ≥ ${usd(CONFIG.MIN_POSITION_USD)}, ` +
      `up to ${CONFIG.MAX_POSITIONS_PER_USER} per user (largest first).`
  );
  lines.push('');
  lines.push(`Total performers: ${users.length}`);
  lines.push('');
  lines.push('---');

  users.forEach((u, i) => {
    const rankTags = Object.entries(u.boards)
      .map(([label, v]) => `${label} #${v.rank}`)
      .join(' · ');
    lines.push('');
    lines.push(`## ${i + 1}. ${u.name}`);
    lines.push('');
    lines.push(`- Wallet: \`${u.wallet}\``);
    lines.push(`- Profile: https://polymarket.com/profile/${u.wallet}`);
    lines.push(`- Leaderboard: ${rankTags}`);
    lines.push(
      `- Tracked open positions: ${u.positions.length} · ` +
        `Portfolio (tracked) value: ${usd(u.totalValue)} · ` +
        `Unrealized PnL: ${usd(u.totalPnl)}`
    );
    lines.push('');
    if (!u.positions.length) {
      lines.push('_No open positions above threshold._');
      return;
    }
    for (const p of u.positions) {
      lines.push(`### ${p.title}`);
      lines.push(
        `${p.outcome} ${cents(p.curPrice)} · ${shares(p.size)} shares · ` +
          `avg ${cents(p.avgPrice)} → cur ${cents(p.curPrice)}`
      );
      lines.push(
        `${usd(p.currentValue)} · PnL ${usd(p.cashPnl)} (${pct(p.percentPnl)})` +
          (p.endDate ? ` · ends ${String(p.endDate).slice(0, 10)}` : '')
      );
      lines.push('');
    }
    lines.push('---');
  });
  return lines.join('\n');
}

function renderCommon(users, date) {
  // Aggregate identical market+outcome holdings across the pool.
  const map = new Map();
  for (const u of users) {
    for (const p of u.positions) {
      const key = `${p.conditionId}::${p.outcome}`;
      if (!map.has(key)) {
        map.set(key, {
          title: p.title,
          outcome: p.outcome,
          slug: p.slug,
          holders: [],
          totalShares: 0,
          totalValue: 0,
          curPrice: p.curPrice,
          endDate: p.endDate,
          avgPriceSum: 0,
        });
      }
      const e = map.get(key);
      e.holders.push({ name: u.name, value: p.currentValue, size: p.size, avgPrice: p.avgPrice });
      e.totalShares += p.size;
      e.totalValue += p.currentValue;
      e.avgPriceSum += p.avgPrice;
      e.curPrice = p.curPrice; // latest
    }
  }
  const rows = [...map.values()]
    .map((e) => ({ ...e, holderCount: e.holders.length, avgEntry: e.avgPriceSum / e.holders.length }))
    .filter((e) => e.holderCount >= 2)
    .sort((a, b) => b.holderCount - a.holderCount || b.totalValue - a.totalValue);

  const lines = [];
  lines.push(`# Polymarket Top Performers — Common Positions`);
  lines.push('');
  lines.push(`Generated: ${date}`);
  lines.push('');
  lines.push(
    `Markets/outcomes held by ≥ 2 of the top ${users.length} performers, ` +
      `ranked by number of holders then aggregate value. This is the basket-copy signal: ` +
      `positions many strong traders share.`
  );
  lines.push('');
  lines.push(`| # | Holders | Market | Outcome | Cur Price | Avg Entry | Total Shares | Aggregate Value |`);
  lines.push(`|---|--------:|--------|---------|----------:|----------:|-------------:|----------------:|`);
  rows.forEach((e, i) => {
    lines.push(
      `| ${i + 1} | ${e.holderCount} | ${e.title.replace(/\|/g, '\\|')} | ${e.outcome} | ` +
        `${cents(e.curPrice)} | ${cents(e.avgEntry)} | ${shares(e.totalShares)} | ${usd(e.totalValue)} |`
    );
  });
  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('## Holder detail (top 40 shared positions)');
  rows.slice(0, 40).forEach((e, i) => {
    lines.push('');
    lines.push(`### ${i + 1}. ${e.title} — ${e.outcome} (${e.holderCount} holders)`);
    e.holders
      .sort((a, b) => b.value - a.value)
      .forEach((h) => lines.push(`- ${h.name}: ${shares(h.size)} shares @ avg ${cents(h.avgPrice)} = ${usd(h.value)}`));
  });
  return lines.join('\n');
}

// --------------------------------- MAIN --------------------------------------
async function main() {
  const date = today();
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });

  const poolUsers = await buildUserPool();
  if (!poolUsers.length) {
    console.error('No users found — leaderboard API may be down. Aborting.');
    process.exit(1);
  }

  console.log(`\n[2/3] Fetching open positions for ${poolUsers.length} users (concurrency ${CONFIG.CONCURRENCY})...`);
  const enriched = await pool(
    poolUsers,
    CONFIG.CONCURRENCY,
    (u) => fetchUserPositions(u),
    (done, total) => process.stdout.write(`\r  ${done}/${total} users done`)
  );
  process.stdout.write('\n');

  const users = enriched.filter((u) => u && !u.error);
  const failed = enriched.filter((u) => u && u.error);
  if (failed.length) console.warn(`  ! ${failed.length} users failed to fetch.`);

  console.log(`\n[3/3] Writing outputs to ${CONFIG.OUTPUT_DIR} ...`);
  const mdPath = path.join(CONFIG.OUTPUT_DIR, `top100_positions_${date}.md`);
  const jsonPath = path.join(CONFIG.OUTPUT_DIR, `top100_positions_${date}.json`);
  const commonPath = path.join(CONFIG.OUTPUT_DIR, `common_positions_${date}.md`);

  fs.writeFileSync(mdPath, renderMarkdown(users, date), 'utf8');
  fs.writeFileSync(
    jsonPath,
    JSON.stringify(
      {
        generated: date,
        config: CONFIG,
        userCount: users.length,
        users: users.map((u, i) => ({
          rank: i + 1,
          name: u.name,
          wallet: u.wallet,
          boards: u.boards,
          totalValue: u.totalValue,
          totalPnl: u.totalPnl,
          positions: u.positions,
        })),
      },
      null,
      2
    ),
    'utf8'
  );
  fs.writeFileSync(commonPath, renderCommon(users, date), 'utf8');

  const totalPositions = users.reduce((s, u) => s + u.positions.length, 0);
  console.log(`\n✓ Done.`);
  console.log(`  Users: ${users.length} · Total tracked open positions: ${totalPositions}`);
  console.log(`  ${mdPath}`);
  console.log(`  ${jsonPath}`);
  console.log(`  ${commonPath}`);
}

main().catch((err) => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
