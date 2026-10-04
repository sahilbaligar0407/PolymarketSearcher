'use strict';
const $ = (id) => document.getElementById(id);
const usd = (n) => (n < 0 ? '-$' : '$') + Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 0 });
const usd2 = (n) => '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const cents = (p) => (p * 100).toFixed(1) + '¢';
const num = (n) => Number(n).toLocaleString('en-US', { maximumFractionDigits: 1 });

function status(msg, spin) {
  $('statusMsg').innerHTML = (spin ? '<span class="spinner"></span> ' : '') + msg;
}

// remove browser-load transition guard after first paint
window.addEventListener('load', () => setTimeout(() => document.body.classList.remove('no-transition'), 60));

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

// ---------- render dashboard ----------
function renderDashboard(d) {
  $('mPool').textContent = d.userCount;
  $('mSignals').textContent = d.common.length;
  $('mDate').textContent = d.generated || '—';
  $('kTraders').textContent = d.userCount;
  $('kCapital').textContent = usd(d.totalTrackedValue);
  const pnl = $('kPnl');
  pnl.textContent = usd(d.totalTrackedPnl);
  pnl.className = 'kpi-val ' + (d.totalTrackedPnl >= 0 ? 'pos' : 'neg');
  $('kSignals').textContent = d.common.length;
  $('kTop').textContent = d.common[0] ? `${d.common[0].holderCount}× ${d.common[0].title.slice(0, 40)}` : '—';

  // consensus signal table
  const tb = $('signalTbl').querySelector('tbody');
  tb.innerHTML = '';
  d.common.forEach((c, i) => {
    const tr = document.createElement('tr');
    const sideCls = /yes|over|under|^no$/i.test(c.outcome) ? '' : '';
    tr.innerHTML =
      `<td class="num">${i + 1}</td>` +
      `<td><span class="hold-pill">${c.holderCount}</span></td>` +
      `<td>${esc(c.title)}</td>` +
      `<td>${esc(c.outcome)}</td>` +
      `<td class="num">${cents(c.curPrice)}</td>` +
      `<td class="num">${cents(c.avgEntry)}</td>` +
      `<td class="num">${usd(c.totalValue)}</td>`;
    tb.appendChild(tr);
  });
  $('signalCount').textContent = d.common.length;
  $('signalEmpty').hidden = d.common.length > 0;

  // top traders
  const ol = $('traders');
  ol.innerHTML = '';
  d.topTraders.forEach((t) => {
    const li = document.createElement('li');
    const boards = Object.entries(t.boards || {}).map(([k, v]) => `${k.split(' ')[0]}#${v.rank}`).join(' ');
    li.innerHTML =
      `<span class="t-rank">${t.rank}</span>` +
      `<span><span class="t-name">${esc(t.name)}</span><br><span class="t-sub">${t.positionCount} pos · ${esc(boards)}</span></span>` +
      `<span class="t-val"><span class="v">${usd(t.totalValue)}</span><br><span class="pnl ${t.totalPnl >= 0 ? 'pos' : 'neg'}">${usd(t.totalPnl)}</span></span>`;
    ol.appendChild(li);
  });
  $('traderCount').textContent = d.topTraders.length;
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ---------- allocation ----------
function renderAllocation(alloc, briefing) {
  $('allocEmpty').hidden = true;
  $('allocEngineTag').textContent = briefing.engine;

  const br = $('briefing');
  br.hidden = false;
  br.innerHTML = `<span class="briefing-engine">◆ ANALYST BRIEFING · ${esc(briefing.engine)}</span>${esc(briefing.text)}`;

  const sum = $('allocSummary');
  sum.hidden = false;
  sum.innerHTML =
    `<div class="s"><span>BANKROLL</span><b>${usd2(alloc.budget)}</b></div>` +
    `<div class="s"><span>DEPLOYED</span><b style="color:var(--green)">${usd2(alloc.deployed)}</b></div>` +
    `<div class="s"><span>RESERVED</span><b>${usd2(alloc.reserved)}</b></div>` +
    `<div class="s"><span>POSITIONS</span><b>${alloc.picks.length}</b></div>` +
    `<div class="s"><span>RISK</span><b style="text-transform:capitalize">${esc(alloc.risk)}</b></div>`;

  const box = $('picks');
  box.innerHTML = '';
  if (!alloc.picks.length) {
    box.innerHTML = `<div class="empty">${esc(alloc.note || 'No tradeable candidates.')}</div>`;
    return;
  }
  alloc.picks.forEach((p, i) => {
    const el = document.createElement('div');
    el.className = 'pick';
    const ev = (p.evidence || []).slice(0, 3);
    const evHtml = ev.length
      ? `<div class="evidence"><span class="ev-label">WEB VALIDATION</span>${ev
          .map((e) => `<a href="${esc(e.url)}" target="_blank" rel="noopener">→ ${esc(e.title)}</a>`)
          .join('')}</div>`
      : '';
    el.innerHTML =
      `<div class="p-title">${i + 1}. ${esc(p.title)}</div>` +
      `<div class="p-stake"><div class="amt">${usd2(p.stake)}</div><div class="pct">${p.allocationPct}% of bankroll</div></div>` +
      `<div class="p-meta">BET <span class="p-side">${esc(p.side)}</span> · ` +
      `<span class="p-num"><b>${num(p.shares)}</b> shares</span> · ` +
      `<span class="roi">+${p.roiIfWin}% if it hits</span> · ` +
      `${p.holderCount} traders · ${usd(p.smartMoney)} smart $` +
      (p.endDate ? ` · ends ${String(p.endDate).slice(0, 10)}` : '') + `</div>` +
      `<div class="bar"><i style="width:${Math.min(100, p.allocationPct * 3)}%"></i></div>` +
      `<div class="p-rationale">${esc(p.rationale)}</div>` +
      evHtml;
    box.appendChild(el);
  });
}

// ---------- data flow ----------
let hasData = false;

async function loadData() {
  status('Loading dataset…', true);
  const { ok, data } = await api('/api/data');
  if (!ok) {
    status(data.message || 'No dataset. Click REFRESH DATA.');
    $('signalEmpty').hidden = false;
    return;
  }
  hasData = true;
  renderDashboard(data);
  status(`Loaded ${data.userCount} traders · ${data.common.length} consensus signals · snapshot ${data.generated}`);
}

async function refreshData() {
  const btn = $('refreshBtn');
  btn.disabled = true;
  status('Running collector against Polymarket (10–20s)…', true);
  const { ok, data } = await api('/api/collect', { method: 'POST' });
  btn.disabled = false;
  if (!ok) return status(data.message || 'Collector failed.');
  hasData = true;
  renderDashboard(data);
  status(`Refreshed · ${data.userCount} traders · snapshot ${data.generated}`);
}

async function allocate() {
  if (!hasData) { status('Load or refresh data first.'); return; }
  const btn = $('allocBtn');
  btn.disabled = true;
  const validate = $('validate').checked;
  status(validate ? 'Scoring candidates + web-validating picks…' : 'Scoring candidates…', true);
  const body = { budget: Number($('budget').value), risk: $('risk').value, validate };
  const { ok, data } = await api('/api/allocate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  btn.disabled = false;
  if (!ok) return status(data.message || 'Allocation failed.');
  renderAllocation(data.alloc, data.briefing);
  const eng = data.aiStatus && data.aiStatus.available ? `SLM ${data.aiStatus.model}` : 'heuristic engine';
  status(`Spread ready · ${data.alloc.picks.length} positions · ${usd2(data.alloc.deployed)} deployed · ${eng}`);
}

async function checkAI() {
  const { data } = await api('/api/ai-status');
  $('aiEngine').textContent = data.available ? data.model : 'heuristic';
}

$('refreshBtn').addEventListener('click', refreshData);
$('allocBtn').addEventListener('click', allocate);

checkAI();
loadData();
