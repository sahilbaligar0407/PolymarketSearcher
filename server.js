#!/usr/bin/env node
'use strict';
/*
 * PolymarketSearcher dashboard server. Zero dependencies (Node http + built-ins).
 *
 *   node server.js            -> serves http://localhost:5173
 *
 * Endpoints:
 *   GET  /api/data            latest collected dataset (from output/*.json)
 *   GET  /api/ai-status       is a local SLM (Ollama) available?
 *   POST /api/collect         (re)run the collector, then return fresh data
 *   POST /api/allocate        { budget, risk } -> AI allocation + web validation
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const analyze = require('./lib/analyze');
const websearch = require('./lib/websearch');
const ai = require('./lib/ai');

const PORT = process.env.PORT || 5173;
const ROOT = __dirname;
const PUBLIC = path.join(ROOT, 'public');
const OUTPUT = path.join(ROOT, 'output');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function send(res, code, body, headers = {}) {
  res.writeHead(code, { 'Cache-Control': 'no-store', ...headers });
  res.end(body);
}
function sendJSON(res, code, obj) {
  send(res, code, JSON.stringify(obj), { 'Content-Type': 'application/json; charset=utf-8' });
}

// Find the newest top100_positions_*.json in output/.
function latestDataset() {
  if (!fs.existsSync(OUTPUT)) return null;
  const files = fs
    .readdirSync(OUTPUT)
    .filter((f) => /^top100_positions_.*\.json$/.test(f))
    .map((f) => ({ f, t: fs.statSync(path.join(OUTPUT, f)).mtimeMs }))
    .sort((a, b) => b.t - a.t);
  if (!files.length) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(OUTPUT, files[0].f), 'utf8'));
    return { file: files[0].f, data: raw };
  } catch {
    return null;
  }
}

function readBody(req) {
  return new Promise((resolve) => {
    let b = '';
    req.on('data', (c) => (b += c));
    req.on('end', () => {
      try {
        resolve(b ? JSON.parse(b) : {});
      } catch {
        resolve({});
      }
    });
  });
}

function runCollector() {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(ROOT, 'collect.js')], { cwd: ROOT });
    let log = '';
    child.stdout.on('data', (d) => (log += d));
    child.stderr.on('data', (d) => (log += d));
    child.on('close', (code) => (code === 0 ? resolve(log) : reject(new Error('collector exited ' + code))));
    child.on('error', reject);
  });
}

// Build the dashboard payload from a raw dataset (users + positions).
function buildDashboard(raw) {
  const users = raw.users || [];
  const common = analyze.aggregateCommon(users);
  const totalValue = users.reduce((s, u) => s + (u.totalValue || 0), 0);
  const totalPnl = users.reduce((s, u) => s + (u.totalPnl || 0), 0);
  return {
    generated: raw.generated,
    userCount: users.length,
    totalTrackedValue: Math.round(totalValue),
    totalTrackedPnl: Math.round(totalPnl),
    boards: (raw.config && raw.config.BOARDS) || [],
    topTraders: users.slice(0, 100).map((u, i) => ({
      rank: u.rank || i + 1,
      name: u.name,
      wallet: u.wallet,
      boards: u.boards,
      totalValue: Math.round(u.totalValue || 0),
      totalPnl: Math.round(u.totalPnl || 0),
      positionCount: (u.positions || []).length,
    })),
    common: common.filter((c) => c.holderCount >= 2).slice(0, 60).map((c) => ({
      title: c.title,
      outcome: c.outcome,
      curPrice: c.curPrice,
      avgEntry: +c.avgEntry.toFixed(3),
      holderCount: c.holderCount,
      totalShares: Math.round(c.totalShares),
      totalValue: Math.round(c.totalValue),
      endDate: c.endDate,
      holders: c.holders.map((h) => h.name).slice(0, 12),
    })),
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;

  try {
    // -------- API --------
    if (p === '/api/data' && req.method === 'GET') {
      const ds = latestDataset();
      if (!ds) return sendJSON(res, 404, { error: 'no-data', message: 'No dataset yet. Click "Refresh Data" to run the collector.' });
      return sendJSON(res, 200, { file: ds.file, ...buildDashboard(ds.data) });
    }

    if (p === '/api/ai-status' && req.method === 'GET') {
      return sendJSON(res, 200, await ai.status());
    }

    if (p === '/api/collect' && req.method === 'POST') {
      try {
        await runCollector();
        const ds = latestDataset();
        return sendJSON(res, 200, { ok: true, file: ds && ds.file, ...(ds ? buildDashboard(ds.data) : {}) });
      } catch (err) {
        return sendJSON(res, 500, { error: 'collect-failed', message: err.message });
      }
    }

    if (p === '/api/allocate' && req.method === 'POST') {
      const body = await readBody(req);
      const budget = Math.max(1, Number(body.budget) || 100);
      const risk = ['conservative', 'balanced', 'aggressive'].includes(body.risk) ? body.risk : 'balanced';
      const validate = body.validate !== false;

      const ds = latestDataset();
      if (!ds) return sendJSON(res, 400, { error: 'no-data', message: 'Run the collector first.' });

      const common = analyze.aggregateCommon(ds.data.users || []);
      const alloc = analyze.allocate(common, { budget, risk });
      alloc.generatedAt = new Date().toISOString();

      // Web validation for the chosen picks (best-effort, keyless).
      const evidenceByPick = [];
      if (validate) {
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
        // Validate only the top picks to stay under DuckDuckGo's burst limits.
        const VALIDATE_TOP = Math.min(alloc.picks.length, Number(body.validateTop) || 6);
        for (let i = 0; i < alloc.picks.length; i++) {
          const pick = alloc.picks[i];
          if (i >= VALIDATE_TOP) { evidenceByPick.push([]); continue; }
          const q = `${pick.title} prediction odds news`;
          // sequential + gap to stay polite with DuckDuckGo (avoids the challenge page)
          // eslint-disable-next-line no-await-in-loop
          const results = await websearch.search(q, { max: 3 });
          evidenceByPick.push(results);
          pick.evidence = results;
          // eslint-disable-next-line no-await-in-loop
          if (i < VALIDATE_TOP - 1) await sleep(900);
        }
      } else {
        alloc.picks.forEach(() => evidenceByPick.push([]));
      }

      const st = await ai.status();
      const briefing = await ai.generateBriefing(alloc, evidenceByPick, st);
      return sendJSON(res, 200, { alloc, briefing, aiStatus: st });
    }

    // -------- static files --------
    let rel = p === '/' ? '/index.html' : p;
    rel = rel.replace(/\.\./g, '');
    const file = path.join(PUBLIC, rel);
    if (fs.existsSync(file) && fs.statSync(file).isFile()) {
      const ext = path.extname(file);
      return send(res, 200, fs.readFileSync(file), { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    }
    return send(res, 404, 'Not found');
  } catch (err) {
    return sendJSON(res, 500, { error: 'server-error', message: err.message });
  }
});

server.listen(PORT, () => {
  console.log(`\n  PolymarketSearcher dashboard  ->  http://localhost:${PORT}\n`);
  const ds = latestDataset();
  if (ds) console.log(`  Loaded dataset: ${ds.file}`);
  else console.log(`  No dataset yet — open the dashboard and click "Refresh Data".`);
  ai.status().then((s) =>
    console.log(s.available ? `  Local SLM: ${s.model} (Ollama) ✓` : `  Local SLM: none — using heuristic engine (install Ollama to upgrade).`)
  );
});
