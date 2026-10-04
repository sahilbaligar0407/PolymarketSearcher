"""Configuration and secrets.

Secrets come from the environment (``.env``, never committed).  Everything else comes from
layered YAML in ``configs/``.  Nothing in this module ever logs a key.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from marketlab.core.broker import Mode

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"

#: The exact string a user must set to arm live trading. Nothing else works.
LIVE_ARM_TOKEN = "YES_I_ACCEPT_REAL_LOSS"


class Secrets(BaseSettings):
    """Credentials. Absent credentials degrade a source, never crash the engine."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    marketlab_mode: str = "PAPER"

    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_environment: str = "production"

    polymarket_us_key_id: str = ""
    polymarket_us_secret_key: str = ""

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    x_bearer_token: str = ""
    the_odds_api_key: str = ""
    fred_api_key: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""

    local_llm_provider: str = "auto"
    local_llm_base_url: str = ""
    local_llm_model: str = ""

    #: Remote second-opinion tier. Hard-capped per UTC day; 0 disables it entirely.
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-nano"
    openai_daily_budget_usd: str = "0.02"

    #: Jev: any OpenAI-compatible endpoint. Unset URL => no Jev arms are created.
    jev_base_url: str = ""
    jev_api_key: str = ""
    jev_model: str = ""

    live_trading_enabled: str = "NO"

    #: SEC rejects any User-Agent lacking a contact email (its bot filter 403s on the
    #: missing "@"). The default is deliberately unusable so the failure is explicit.
    sec_user_agent: str = "MarketLab research (set SEC_USER_AGENT to an email)"

    @property
    def live_armed(self) -> bool:
        return self.live_trading_enabled.strip() == LIVE_ARM_TOKEN

    def redacted(self) -> dict[str, str]:
        """Presence map for `marketlab doctor`. Values are never included."""
        out: dict[str, str] = {}
        for name in type(self).model_fields:
            if name in {
                "marketlab_mode", "local_llm_provider", "kalshi_environment",
                "openai_model", "openai_daily_budget_usd", "jev_model",
            }:
                out[name] = str(getattr(self, name))
                continue
            value = str(getattr(self, name) or "")
            out[name] = "SET" if value and value != "NO" else "MISSING"
        return out


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_to_order_ms: int = 100
    network_latency_ms: int = 75
    processing_latency_ms: int = 25
    #: TOUCH | TRADE_THROUGH | QUEUE
    limit_fill_model: str = "TRADE_THROUGH"
    #: Books older than this are refused; the engine does not trade blind.
    max_book_age_seconds: float = 30.0
    #: Extra probability subtracted from every claimed edge to cover bad fills.
    slippage_buffer: Decimal = Decimal("0.005")
    uncertainty_buffer: Decimal = Decimal("0.01")
    #: Latency arms swept by the experiment generator, in milliseconds.
    latency_sweep_ms: tuple[int, ...] = (50, 100, 250, 500, 1000, 5000, 30000)

    @property
    def total_latency_ms(self) -> int:
        return self.signal_to_order_ms + self.network_latency_ms + self.processing_latency_ms


class RiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_capital: Decimal = Decimal("50.00")
    leverage: bool = False
    borrowing: bool = False
    martingale: bool = False
    max_single_event_loss_pct: Decimal = Decimal("0.04")
    max_strategy_exposure_pct: Decimal = Decimal("0.20")
    max_category_exposure_pct: Decimal = Decimal("0.30")
    max_correlated_cluster_pct: Decimal = Decimal("0.30")
    daily_loss_pause_pct: Decimal = Decimal("0.10")
    total_drawdown_pause_pct: Decimal = Decimal("0.20")
    auto_replenish: bool = False
    #: Sleeve is DEAD below this equity.
    death_floor: Decimal = Decimal("1.00")
    #: Reject intents with no rationale.
    strict_audit: bool = True


class PaperConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bankroll_per_strategy: Decimal = Decimal("50.00")
    #: Seconds between engine ticks.
    tick_seconds: float = 1.0
    #: Seconds between leaderboard/report snapshots.
    snapshot_seconds: float = 300.0
    max_concurrent_strategies: int = 400


class SourcesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kalshi_rest: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_rest_fallback: str = "https://external-api.kalshi.com/trade-api/v2"
    kalshi_ws: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_demo_rest: str = "https://demo-api.kalshi.co/trade-api/v2"
    kalshi_demo_ws: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    poly_gamma: str = "https://gamma-api.polymarket.com"
    poly_clob: str = "https://clob.polymarket.com"
    poly_data: str = "https://data-api.polymarket.com"
    poly_leaderboard: str = "https://data-api.polymarket.com/v1/leaderboard"
    poly_lb_legacy: str = "https://lb-api.polymarket.com"
    poly_geoblock: str = "https://polymarket.com/api/geoblock"
    poly_us_rest: str = "https://api.polymarket.us"
    gdelt_doc: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    gdelt_context: str = "https://api.gdeltproject.org/api/v2/context/context"
    sec_base: str = "https://data.sec.gov"
    sec_edgar: str = "https://www.sec.gov"
    fred_base: str = "https://api.stlouisfed.org/fred"
    nws_base: str = "https://api.weather.gov"
    odds_base: str = "https://api.the-odds-api.com/v4"
    ollama_base: str = "http://localhost:11434"
    coinbase_spot: str = "https://api.exchange.coinbase.com"
    binance_spot: str = "https://api.binance.com"
    bluesky_firehose: str = "wss://jetstream2.us-east.bsky.network/subscribe"
    alpaca_data: str = "https://data.alpaca.markets"


class Settings(BaseModel):
    """Everything the engine needs, assembled from env + YAML."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    mode: Mode = Mode.PAPER
    secrets: Secrets = Field(default_factory=Secrets)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    universes: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] = Field(default_factory=dict)
    copy_traders: dict[str, Any] = Field(default_factory=dict)
    source_toggles: dict[str, Any] = Field(default_factory=dict)

    #: Raw top-level blocks from default.yaml/<profile>.yaml that have no typed model.
    #: They are surfaced here rather than being silently dropped - pydantic discards
    #: unknown kwargs, so anything not declared as a field is invisible to callers.
    ingest: dict[str, Any] = Field(default_factory=dict)
    ai: dict[str, Any] = Field(default_factory=dict)
    logging_config: dict[str, Any] = Field(default_factory=dict)

    #: Resolved at startup by the Polymarket adapter; can only ever be set to False.
    polymarket_global_execution: bool = False

    data_dir: Path = DATA_DIR
    db_path: Path = DATA_DIR / "marketlab.db"
    parquet_dir: Path = DATA_DIR / "parquet"
    reports_dir: Path = DATA_DIR / "reports"
    log_dir: Path = DATA_DIR / "logs"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.parquet_dir,
            self.reports_dir,
            self.log_dir,
            self.data_dir / "raw",
            self.data_dir / "normalized",
        ):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def live_allowed(self) -> bool:
        """Live requires the arm token AND the mode AND a credential. Any gap means no."""
        return (
            self.mode is Mode.LIVE
            and self.secrets.live_armed
            and bool(self.secrets.kalshi_api_key_id)
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(profile: str | None = None, config_dir: Path | None = None) -> Settings:
    """Load ``default.yaml`` then overlay ``<profile>.yaml`` and the split config files."""
    cdir = config_dir or CONFIG_DIR
    merged = _load_yaml(cdir / "default.yaml")
    profile = profile or os.getenv("MARKETLAB_PROFILE", "paper")
    merged = _deep_merge(merged, _load_yaml(cdir / f"{profile}.yaml"))

    secrets = Secrets()
    mode_name = os.getenv("MARKETLAB_MODE", merged.get("mode", secrets.marketlab_mode))
    try:
        mode = Mode(str(mode_name).upper())
    except ValueError:
        mode = Mode.PAPER

    risk_yaml = _load_yaml(cdir / "risk.yaml")
    risk_block = risk_yaml.get("live", merged.get("risk", {}))

    settings = Settings(
        mode=mode,
        secrets=secrets,
        execution=ExecutionConfig(**merged.get("execution", {})),
        risk=RiskConfig(**risk_block),
        paper=PaperConfig(**merged.get("paper", {})),
        sources=SourcesConfig(**merged.get("sources", {})),
        universes=_load_yaml(cdir / "universes.yaml"),
        strategies=_load_yaml(cdir / "strategies.yaml"),
        copy_traders=_load_yaml(cdir / "copy_traders.yaml"),
        source_toggles=_load_yaml(cdir / "sources.yaml"),
        ingest=dict(merged.get("ingest") or {}),
        ai=dict(merged.get("ai") or {}),
        logging_config=dict(merged.get("logging") or {}),
    )
    settings.ensure_dirs()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
