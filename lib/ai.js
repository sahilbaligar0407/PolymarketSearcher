'use strict';
/*
 * Local AI layer. Auto-detects an Ollama server at localhost:11434. When present,
 * it writes an "analyst briefing" over the deterministic allocation (the numbers
 * always come from lib/analyze.js so they're never hallucinated — the SLM only
 * adds qualitative reasoning + a risk read, informed by web-search headlines).
 *
 * When Ollama is absent, generateBriefing() returns a solid heuristic briefing so
 * the product is fully functional with zero setup.
 */

const OLLAMA = process.env.OLLAMA_HOST || 'http://localhost:11434';
// Prefer small, fast, instruct-tuned models if the user has any of these pulled.
const PREFERRED = ['llama3.2', 'qwen2.5:3b', 'qwen2.5', 'phi3.5', 'phi3', 'llama3.1', 'mistral', 'gemma2'];

async function status() {
  try {
    const res = await fetch(`${OLLAMA}/api/tags`, { signal: AbortSignal.timeout(2500) });
    if (!res.ok) return { available: false };
    const data = await res.json();
    const models = (data.models || []).map((m) => m.name);
    return { available: models.length > 0, models, host: OLLAMA, model: pickModel(models) };
  } catch {
    return { available: false };
  }
}

function pickModel(models) {
  for (const pref of PREFERRED) {
    const found = models.find((m) => m.startsWith(pref));
    if (found) return found;
  }
  return models[0] || null;
}

async function ollamaChat(model, system, user) {
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      options: { temperature: 0.4 },
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
  const data = await res.json();
  return data.message?.content?.trim() || '';
}

// Compact the allocation + web evidence into a prompt.
function buildPrompt(alloc, evidenceByPick) {
  const lines = [];
  lines.push(`Bankroll: $${alloc.budget}. Risk profile: ${alloc.risk}. Deploying $${alloc.deployed} across ${alloc.picks.length} positions.`);
  lines.push('');
  alloc.picks.forEach((p, i) => {
    lines.push(`${i + 1}. ${p.title} — bet ${p.side}. Stake $${p.stake} (${p.allocationPct}% of bankroll), ${p.shares} shares, +${p.roiIfWin}% if it resolves your way.`);
    lines.push(`   Smart money: ${p.holderCount} top traders, $${p.smartMoney.toLocaleString()} committed. ${p.rationale}`);
    const ev = evidenceByPick[i] || [];
    if (ev.length) {
      lines.push(`   Recent web results:`);
      ev.slice(0, 3).forEach((e) => lines.push(`     - ${e.title}${e.snippet ? ': ' + e.snippet : ''}`));
    }
    lines.push('');
  });
  return lines.join('\n');
}

const SYSTEM = `You are a sharp prediction-market analyst. You are given a pre-computed bankroll allocation across Polymarket positions that top traders hold, plus recent web headlines. Do NOT change the stake amounts. Your job: (1) write a 2-3 sentence overall briefing on the strategy and its balance of risk, (2) for each numbered position give ONE short line flagging whether recent news supports or threatens the bet. Be concrete, skeptical, and concise. No markdown headers, no preamble.`;

async function generateBriefing(alloc, evidenceByPick, st) {
  if (st && st.available && st.model) {
    try {
      const text = await ollamaChat(st.model, SYSTEM, buildPrompt(alloc, evidenceByPick));
      if (text) return { engine: `ollama:${st.model}`, text };
    } catch (err) {
      // fall through to heuristic
    }
  }
  return { engine: 'heuristic', text: heuristicBriefing(alloc, evidenceByPick) };
}

function heuristicBriefing(alloc, evidenceByPick) {
  const avgConv = alloc.picks.reduce((s, p) => s + p.holderCount, 0) / (alloc.picks.length || 1);
  const favorites = alloc.picks.filter((p) => p.curPrice > 0.7).length;
  const coinflips = alloc.picks.filter((p) => p.curPrice > 0.35 && p.curPrice < 0.65).length;
  const lines = [];
  lines.push(
    `This ${alloc.risk} spread deploys $${alloc.deployed} of your $${alloc.budget} across ${alloc.picks.length} positions ` +
      `(holding $${alloc.reserved} in reserve), each backed by an average of ${avgConv.toFixed(0)} top traders. ` +
      `The book leans on ${favorites} high-confidence favorite(s) and ${coinflips} balanced coin-flip(s) for upside — ` +
      `weighting toward consensus among proven traders while capping single-position risk.`
  );
  lines.push('');
  alloc.picks.forEach((p, i) => {
    const ev = evidenceByPick[i] || [];
    let read;
    if (ev.length) {
      read = `web check found "${ev[0].title.slice(0, 90)}" — review for confirmation/contradiction before sizing up.`;
    } else {
      read = `no fresh headlines pulled — rely on the ${p.holderCount}-trader consensus and $${p.smartMoney.toLocaleString()} committed.`;
    }
    lines.push(`${i + 1}. ${p.outcome} on "${p.title.slice(0, 60)}": ${read}`);
  });
  return lines.join('\n');
}

module.exports = { status, generateBriefing };
