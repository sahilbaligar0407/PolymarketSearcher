"""TraderScore: rank Polymarket wallets by their *realized* track record.

The public leaderboard ranks by P&L, which rewards size and one lucky hit as much as
skill. This module scores a wallet from its closed positions (Data API
``/closed-positions``: realized P&L, stake and entry price per resolved bet) and is
explicit about the shapes that look like edge but are not:

* **one lucky trade** - a single position carrying most of the profit;
* **favorite farming** - buying at 0.95+ wins ~95% of the time and looks superb until one
  upset erases fifty wins, so win rate alone is never the score;
* **short history** - fewer than ~30 resolved bets is an anecdote;
* **fading form** - lifetime profit with a losing last 30 days.

Deterministic: same positions in, same score out. Point-in-time: a score computed at T
uses only positions closed by T, and is only ever used to choose whom to follow *after*
T, never to backtest the wallet's past (the copy arms are forward tests).

We never label a wallet "$1 -> $10K" or similar: starting capital is not observable from
public data, so only the measurable statistics below are reported.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

#: Thresholds for promotion to QUALIFIED (copyable). Deliberately strict: the copy arms
#: are judged on whether following these wallets pays on Kalshi, so a loose roster only
#: dilutes the experiment.
MIN_RESOLVED_FOR_QUALIFIED = 30
MIN_SCORE_FOR_QUALIFIED = 70.0
#: Each time-half of the record must clear this ROI on its own (see split-half below).
MIN_HALF_ROI = 0.01
MAX_SCORE_FOR_REJECTED = 35.0
LUCKY_SHARE = 0.5
FAVORITE_PRICE = 0.90


@dataclass(frozen=True)
class TraderAnalytics:
    wallet: str
    username: str
    computed_at: datetime
    resolved_positions: int
    realized_pnl: float
    total_staked: float
    roi: float
    win_rate: float
    profit_factor: float | None
    max_drawdown: float
    sharpe_like: float
    trades_per_day: float
    avg_entry_price: float
    favorite_share: float
    largest_win_share: float
    recent_roi: float | None
    recent_positions: int
    top_category: str
    category_share: float
    score: float
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = "; ".join(self.reasons)
        row["computed_at"] = self.computed_at.isoformat()
        return row


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _category(slug: str) -> str:
    return (slug or "other").split("-", 1)[0] or "other"


def analyze_closed_positions(
    wallet: str, rows: list[dict[str, Any]], now: datetime, username: str = ""
) -> TraderAnalytics:
    """Score one wallet from its closed positions (newest or oldest first, either)."""
    positions = []
    for r in rows:
        ts = r.get("timestamp")
        try:
            closed_at = datetime.fromtimestamp(int(ts), UTC) if ts else None
        except (TypeError, ValueError, OSError):
            closed_at = None
        if closed_at is not None and closed_at > now:
            continue  # never let a "future" record in
        stake = _f(r.get("totalBought")) * _f(r.get("avgPrice")) or _f(r.get("totalBought"))
        positions.append({
            "pnl": _f(r.get("realizedPnl")),
            "stake": max(stake, 1e-9),
            "price": _f(r.get("avgPrice")),
            "closed_at": closed_at or now,
            "category": _category(str(r.get("slug") or r.get("eventSlug") or "")),
        })
    positions.sort(key=lambda p: p["closed_at"])
    n = len(positions)
    reasons: list[str] = []
    if n == 0:
        return TraderAnalytics(
            wallet, username, now, 0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            None, 0, "other", 0.0, 0.0, "DISCOVERED", ("no resolved positions",),
        )

    pnls = [p["pnl"] for p in positions]
    stakes = [p["stake"] for p in positions]
    realized = sum(pnls)
    staked = sum(stakes)
    roi = realized / staked if staked > 0 else 0.0
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    win_rate = len(wins) / n
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else None

    peak = cum = 0.0
    max_dd = 0.0
    for x in pnls:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    returns = [p["pnl"] / p["stake"] for p in positions]
    mean_r = sum(returns) / n
    sd_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / (n - 1)) if n > 1 else 0.0
    sharpe_like = (mean_r / sd_r) * math.sqrt(n) if sd_r > 0 else 0.0

    span_days = max((positions[-1]["closed_at"] - positions[0]["closed_at"]).total_seconds() / 86400, 1.0)
    avg_price = sum(p["price"] for p in positions) / n
    favorite_share = sum(1 for p in positions if p["price"] >= FAVORITE_PRICE) / n
    largest_win_share = (max(wins) / sum(wins)) if wins and realized > 0 else 0.0

    recent = [p for p in positions if p["closed_at"] >= now - timedelta(days=30)]
    recent_staked = sum(p["stake"] for p in recent)
    recent_roi = (sum(p["pnl"] for p in recent) / recent_staked) if recent and recent_staked > 0 else None

    # Split-half consistency. Leaderboard wallets are selected *because* their recent
    # record is good, so a strong aggregate can be one hot streak. Requiring the older and
    # newer halves to be profitable independently is a cheap guard against that.
    half = n // 2
    def _roi(ps: list[dict[str, Any]]) -> float:
        st = sum(p["stake"] for p in ps)
        return sum(p["pnl"] for p in ps) / st if st > 0 else 0.0
    early_roi, late_roi = (_roi(positions[:half]), _roi(positions[half:])) if half else (roi, roi)
    consistent = early_roi > MIN_HALF_ROI and late_roi > MIN_HALF_ROI

    cats = Counter(p["category"] for p in positions)
    top_cat, top_count = cats.most_common(1)[0]

    # ---- score: 50 neutral, evidence moves it --------------------------------
    score = 50.0
    score += max(-25.0, min(25.0, roi * 250))  # +25 at 10% ROI on stake
    score += max(-10.0, min(15.0, sharpe_like * 3))  # consistency, sample-aware
    if profit_factor is not None:
        score += max(-10.0, min(10.0, (profit_factor - 1.0) * 10))
    elif wins:
        score += 5.0  # no losses at all yet: good, but not proof
    if n < MIN_RESOLVED_FOR_QUALIFIED:
        score -= (MIN_RESOLVED_FOR_QUALIFIED - n) * 0.8
        reasons.append(f"short history: {n} resolved positions")
    if largest_win_share > LUCKY_SHARE:
        score -= (largest_win_share - LUCKY_SHARE) * 40
        reasons.append(f"one position is {largest_win_share:.0%} of all winnings (luck risk)")
    if favorite_share > 0.6:
        score -= (favorite_share - 0.6) * 25
        reasons.append(f"{favorite_share:.0%} of bets bought at >= {FAVORITE_PRICE:.2f} (favorite farming)")
    if recent_roi is not None and recent_roi < 0 and roi > 0:
        score -= min(10.0, abs(recent_roi) * 100)
        reasons.append(f"last-30-day ROI {recent_roi:+.1%} while lifetime is positive (fading)")
    if realized > 0 and max_dd > realized:
        score -= 5.0
        reasons.append("drawdown exceeds total profit")
    score = round(max(0.0, min(100.0, score)), 1)

    if (
        score >= MIN_SCORE_FOR_QUALIFIED
        and n >= MIN_RESOLVED_FOR_QUALIFIED
        and roi > 0.01
        and largest_win_share <= LUCKY_SHARE
        and (recent_roi is None or recent_roi > -0.02)
        and consistent
    ):
        status = "QUALIFIED"
        reasons.insert(0, f"qualified: ROI {roi:+.1%} over {n} resolved bets")
    elif score <= MAX_SCORE_FOR_REJECTED:
        status = "REJECTED"
    elif not consistent and score >= MIN_SCORE_FOR_QUALIFIED:
        reasons.append(f"inconsistent halves: early ROI {early_roi:+.1%}, late ROI {late_roi:+.1%}")
        status = "TRACKING"
    else:
        status = "TRACKING"

    return TraderAnalytics(
        wallet=wallet, username=username, computed_at=now, resolved_positions=n,
        realized_pnl=round(realized, 2), total_staked=round(staked, 2), roi=round(roi, 4),
        win_rate=round(win_rate, 4), profit_factor=round(profit_factor, 3) if profit_factor is not None else None,
        max_drawdown=round(max_dd, 2), sharpe_like=round(sharpe_like, 3),
        trades_per_day=round(n / span_days, 3), avg_entry_price=round(avg_price, 4),
        favorite_share=round(favorite_share, 3), largest_win_share=round(largest_win_share, 3),
        recent_roi=round(recent_roi, 4) if recent_roi is not None else None, recent_positions=len(recent),
        top_category=top_cat, category_share=round(top_count / n, 3), score=score, status=status,
        reasons=tuple(reasons),
    )


def median_score(items: list[TraderAnalytics]) -> float:
    return median([i.score for i in items]) if items else 0.0


__all__ = ["TraderAnalytics", "analyze_closed_positions"]
