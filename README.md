# PolymarketSearcher

Collects the **open positions of the top ~100 Polymarket performers** into one mega
document, plus a cross-trader "what do the top performers hold in common" table for
basket-copy analysis. Pure public API, **no browser, no login, zero dependencies.**

## Run it

Requires Node 18+ (tested on Node 24). No `npm install` needed.

```powershell
node collect.js
```

or `npm start`, or double-click `run.bat`.

Takes ~10–15 seconds. Writes three dated files into `output/`:

| File | What it is |
|------|-----------|
| `top100_positions_<date>.md`   | Human-readable mega doc — every top performer with their open positions in the format `Outcome ¢ · shares · avg→cur · $value · PnL`. Upload this (or the JSON) to ChatGPT. |
| `top100_positions_<date>.json` | Same data, structured, for programmatic / ChatGPT use. |
| `common_positions_<date>.md`   | **Basket-copy signal:** markets/outcomes held by ≥2 top performers, ranked by holder count + aggregate value, with per-holder detail. |

## How it works

1. **Top-performer pool.** The public leaderboard API (`lb-api.polymarket.com`) hard-caps
   each board at the top 50 and does **not** support offset/pagination. To reach ~100
   unique performers we union four boards — Monthly Profit, Monthly Volume, All-time
   Profit, All-time Volume — deduping by wallet and tagging each trader with every rank
   they hold. Monthly Profit (the site default) defines the primary ordering.
2. **Positions.** For each wallet we call the public positions API
   (`data-api.polymarket.com/positions`), sorted by current value descending, and keep
   the meaningful **open** positions (unresolved, `0 < price < 1`, value ≥ threshold).
3. **Render** the three documents above.

## Tuning

Edit the `CONFIG` block at the top of `collect.js`:

| Key | Default | Meaning |
|-----|---------|---------|
| `TARGET_USERS` | `100` | Size of the performer pool. |
| `BOARDS` | 4 boards | Which leaderboards to union, and their priority (first = primary rank). Each: `{ metric: 'profit'\|'volume', window: '1d'\|'7d'\|'30d'\|'all', label }`. |
| `MIN_POSITION_USD` | `50` | Ignore open positions worth less than this. |
| `MAX_POSITIONS_PER_USER` | `40` | Keep at most this many (largest) per user. |
| `CONCURRENCY` | `6` | Parallel position requests. |

### Want strictly "Monthly Profit only"?

Set `BOARDS` to just `[{ metric: 'profit', window: '30d', label: 'Monthly Profit' }]`.
You'll get the true top **50** by monthly profit (the public API can't return ranks 51–100
for a single board).

## Notes

- Positions valued near 99–100¢ are effectively settled — the collector still includes them
  (matching what you see on a profile). Your downstream analysis / ChatGPT step decides which
  positions still have room to copy.
- The `output/` folder is git-ignored so runs don't clutter version control.
