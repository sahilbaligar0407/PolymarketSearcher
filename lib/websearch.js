'use strict';
/*
 * Keyless web validation via DuckDuckGo. Best-effort: DuckDuckGo throttles
 * automated/repeated requests, so on failure this returns an empty list rather
 * than throwing, and the caller degrades gracefully.
 *
 * Strategy: POST the query (DDG's real form method) to the html endpoint, then
 * fall back to the lite endpoint. Results are cached for 30 min.
 */

const cache = new Map();
const TTL_MS = 30 * 60 * 1000;
const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36';

function decode(s) {
  return String(s)
    .replace(/<[^>]*>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function unwrap(link) {
  const m = link.match(/uddg=([^&]+)/);
  if (m) {
    try { return decodeURIComponent(m[1]); } catch { /* ignore */ }
  }
  return link.startsWith('//') ? 'https:' + link : link;
}

async function tryHtml(query, max) {
  const res = await fetch('https://html.duckduckgo.com/html/', {
    method: 'POST',
    headers: { 'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ q: query, kl: 'us-en' }).toString(),
    signal: AbortSignal.timeout(12000),
  });
  const html = await res.text();
  const out = [];
  const re = /result__a"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>[\s\S]*?result__snippet"[^>]*>(.*?)<\/a>/g;
  let m;
  while ((m = re.exec(html)) && out.length < max) {
    out.push({ title: decode(m[2]), snippet: decode(m[3]).slice(0, 220), url: unwrap(m[1]) });
  }
  return out;
}

async function tryLite(query, max) {
  const res = await fetch('https://lite.duckduckgo.com/lite/', {
    method: 'POST',
    headers: { 'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ q: query, kl: 'us-en' }).toString(),
    signal: AbortSignal.timeout(12000),
  });
  const html = await res.text();
  const out = [];
  const re = /class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>/g;
  let m;
  while ((m = re.exec(html)) && out.length < max) {
    out.push({ title: decode(m[2]), snippet: '', url: unwrap(m[1]) });
  }
  return out;
}

async function search(query, { max = 3 } = {}) {
  const key = query.toLowerCase();
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.results;

  let results = [];
  try {
    results = await tryHtml(query, max);
    if (!results.length) results = await tryLite(query, max);
  } catch {
    try { results = await tryLite(query, max); } catch { results = []; }
  }
  cache.set(key, { at: Date.now(), results });
  return results;
}

module.exports = { search };
