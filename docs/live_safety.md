# Live trading safety

Live trading is **hard-disabled by default** and is designed to be impossible to activate
by accident. This document is the authoritative description of every gate.

## Venue policy

**Kalshi is the only execution venue.**

Polymarket global is a read-only intelligence source: market prices for cross-venue
comparison, the public leaderboard, and public trader activity. The operator is in the
United States, where global Polymarket is close-only for new positions. MarketLab
therefore:

- never submits an order to Polymarket global, in any mode
- never attempts to bypass, proxy, VPN around, or otherwise defeat the geoblock
- checks the official geoblock endpoint at startup and **fails closed** if it cannot reach it
- sets `settings.polymarket_global_execution = False` and provides no code path that sets it True

A Polymarket price discrepancy is a **signal to trade the equivalent Kalshi contract** —
and only when the matching engine has certified the two markets as the same bet.

Polymarket US is implemented as a public-data adapter only. Authenticated order code is
deliberately absent because it is out of scope for this deployment.

## The four modes

| Mode | Ingests data | Creates orders | Touches a venue |
|---|---|---|---|
| `DATA_ONLY` | yes | no | no |
| `BACKTEST` | replays stored data | simulated | no |
| `PAPER` | live production data | simulated locally | **no create-order call, ever** |
| `LIVE` | live production data | real | yes, Kalshi only |

A strategy cannot tell which mode it is in. It emits `OrderIntent` objects to a `Broker`
interface; the broker decides what happens. That separation exists specifically to prevent
the common failure where the backtested implementation and the live implementation quietly
diverge.

## Everything required to reach LIVE

All of the following must be simultaneously true. Any single gap keeps live disabled.

1. `MARKETLAB_MODE=LIVE` in the environment
2. `LIVE_TRADING_ENABLED=YES_I_ACCEPT_REAL_LOSS` — the exact string; anything else fails
3. Valid Kalshi production credentials (`KALSHI_API_KEY_ID` + a readable private key file)
4. Kalshi environment is `production` and the credentials are production credentials —
   demo credentials are refused against production and vice versa
5. `KalshiLiveBroker` constructed with `acknowledge_real_money_risk=True`
6. The CLI invoked as `marketlab live start --acknowledge-real-money-risk`
7. `marketlab live doctor` reports every check green
8. Full test suite passes
9. Risk config loaded and valid
10. No stale market feed (every required source `HEALTHY`)
11. Venue eligibility check passes
12. The strategy being armed holds status `QUALIFIED` or better

`marketlab live doctor` prints each unmet condition. Live never starts merely because
credentials happen to exist.

## Risk limits on the real $50

```yaml
initial_capital: 50.00
leverage: false
borrowing: false
martingale: false
max_single_event_loss_pct: 0.04     # $2.00
max_strategy_exposure_pct: 0.20     # $10.00
max_category_exposure_pct: 0.30     # $15.00
max_correlated_cluster_pct: 0.30    # $15.00
daily_loss_pause_pct: 0.10          # pause at -$5.00 on the day
total_drawdown_pause_pct: 0.20      # pause at -$10.00 from the high-water mark
auto_replenish: false
```

**If a venue minimum order size would force a trade above a risk limit, the trade is
skipped.** Limits are never relaxed to allow participation.

A daily-loss pause does not falsify the "$50, die with it" experiment. It stops the system
from spending the remainder of the bankroll while in a state we already know is broken,
so that the failure can be diagnosed instead of merely completed.

No automatic Kelly sizing. Fractional-Kelly is computed as a research metric only.
Overconfident probability estimates make Kelly aggressive precisely when it is most wrong,
so real sizing stays hard-capped until calibration has been demonstrated out of sample.

## Capital epochs

The live bankroll may decline to zero. MarketLab will not:

- martingale
- average down automatically because a position moved against it
- borrow or use leverage
- refill the bankroll silently

Adding capital is permitted but is recorded as a **new capital epoch**, so no report can
hide a prior loss behind fresh money.

## The local model's authority

The LLM is an analyst. The enforced pipeline is:

```
DATA -> RETRIEVAL -> LOCAL MODEL -> STRICT JSON -> DETERMINISTIC VALIDATOR
     -> STRATEGY -> RISK GATEWAY -> BROKER
```

The model cannot:

- read private keys or `.env` secrets
- invoke a shell or any tool with side effects
- construct an order
- change a risk limit
- activate live mode
- move funds

Malformed or unvalidatable model output causes an **abstain**. It is never silently
repaired into something tradeable.

## Secret handling

- `.env`, `*.pem`, `*.key`, `credentials*` are gitignored
- the structured logger redacts known secret field names and any PEM block, even if one
  reaches a free-text field
- `marketlab doctor` prints credential **presence**, never values
- signing components are isolated from research and AI code paths

## Data-staleness policy

Every feed tracks `last_message_at`, latency, error count and reconnect count, and reports
`HEALTHY` / `DEGRADED` / `STALE` / `DOWN`. When a required feed goes `STALE`:

- new trading intents are refused
- resting live orders are cancelled where it is safe to do so
- an alert is written

The system does not trade blind.

## First live phase

When something eventually earns promotion:

- total capital $50
- one, or at most a few, `QUALIFIED` strategies
- tiny positions
- no leverage, no martingale, no auto-replenishment
- full audit trail on every decision

Not eighty experimental strategies against the same real bankroll. The experiments stay on
paper. Live is the championship round, and it is meant to be boring.
