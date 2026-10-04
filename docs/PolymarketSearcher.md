# PolymarketSearcher

Tracks the **open positions of the top ~100 Polymarket performers**, finds what the
smart money agrees on, and — given a bankroll — generates an AI **spread of trades**
(which market, which side, how much) validated against live web search.

Pure public API, **no browser, no login, zero npm dependencies.** Runs on Node 18+
(tested on Node 24). Includes a command-center dashboard inspired by
[worldmonitor](https://github.com/koala73/worldmonitor).

---

## Two ways to use it

### 1. The dashboard (recommended)

```powershell
node server.js
```

Then open **http://localhost:5173**. You get:

- **KPI strip** — tracked traders, aggregate capital, unrealized PnL, consensus signals.
- **AI Budget Allocator** — enter a bankroll + risk profile, hit *Generate Spread*, and
  get a diversified set of trades (market, side, stake, shares, ROI-if-win, rationale),
  an **analyst briefing**, and per-pick **web validation** headlines.
- **Consensus Signal** — the markets/outcomes the most top traders share (the basket-copy signal).
- **Top Performers** — the ranked pool with portfolio value + PnL.
- **↻ Refresh Data** — re-runs the collector against Polymarket live.

### 2. The collector (CLI, produces documents)

```powershell
node collect.js       # or: npm run collect
```

Writes three dated files to `output/` (upload the `.md` or `.json` to ChatGPT):

| File | What it is |
|------|-----------|
| `top100_positions_<date>.md`   | Human mega-doc — every top performer + open positions. |
| `top100_positions_<date>.json` | Same, structured, for programmatic/ChatGPT use. |
| `common_positions_<date>.md`   | Cross-trader commonality table (basket-copy signal). |

The dashboard reads the newest `output/*.json`, so run the collector once first (or use
the Refresh Data button).

---

## The AI allocator, explained

The numbers are **deterministic** (computed in `lib/analyze.js`) so they're never
hallucinated; the AI only adds the qualitative layer. For each candidate market held by
≥2 top traders and still tradeable (price between guard rails), it scores four signals:

- **conviction** — how many top traders hold it
- **capital** — aggregate smart-money $ committed (log-scaled)
- **room** — price headroom (peaks at a coin-flip 50¢ — "not yet decided")
- **momentum** — current price vs. the traders' average entry

Risk profile reweights these (conservative favors conviction + likely favorites;
aggressive favors headroom/upside). The bankroll is then spread proportionally to score,
diversified (one pick per event), and capped per trade. Each pick shows stake, shares,
and ROI-if-win.

### Local AI (optional upgrade)

Works out-of-the-box with a built-in **heuristic** analyst. If you install
[Ollama](https://ollama.com) and pull a small model, the app auto-detects it at
`localhost:11434` and upgrades the briefing to real SLM reasoning:

```powershell
# optional
ollama pull llama3.2      # or qwen2.5:3b, phi3.5, etc.
ollama serve
```

The AI badge in the top bar shows the active engine. Point at a different host with
`OLLAMA_HOST`.

### Web validation

Each top pick is checked against **DuckDuckGo** (keyless) for corroborating headlines,
shown inline and fed to the briefing. It's best-effort — DuckDuckGo throttles rapid
automated requests, so only the top picks are validated (paced), results are cached
30 min, and picks with no fresh headlines fall back to the trader-consensus rationale.
Add a real search key later if you want higher reliability.

---

## How the data is built

The public leaderboard API (`lb-api.polymarket.com`) hard-caps each board at the top 50
and does **not** paginate. To reach ~100 unique top performers we union four boards —
Monthly Profit, Monthly Volume, All-time Profit, All-time Volume — deduping by wallet.
Monthly Profit (the site default) defines the primary ordering. Positions come from the
public `data-api.polymarket.com/positions` endpoint (no auth).

## Tuning

Edit `CONFIG` at the top of `collect.js` (`TARGET_USERS`, `BOARDS`, `MIN_POSITION_USD`,
`MAX_POSITIONS_PER_USER`, `CONCURRENCY`). Allocation knobs (`maxPicks`, `maxPerTradePct`,
risk weights) live in `lib/analyze.js`.

## Layout

```
collect.js          CLI collector -> output/ documents
server.js           dashboard server (http://localhost:5173)
lib/analyze.js      consensus aggregation + scoring + allocation (pure)
lib/ai.js           Ollama auto-detect + analyst briefing (heuristic fallback)
lib/websearch.js    keyless DuckDuckGo validation
public/             dashboard UI (index.html, styles.css, app.js)
```

## Disclaimer

For research only. Not financial advice. Copying other traders carries real risk of loss.
