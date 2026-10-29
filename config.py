"""Central configuration, all overridable through environment variables."""
import os
from dataclasses import dataclass


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # Point this at your Week 1-2 multi-tenant DB (e.g. postgresql+psycopg2://user:pw@host/db)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./waste_engine.db")
    model_dir: str = os.getenv("MODEL_DIR", "./model_store")
    api_key: str = os.getenv("API_KEY", "dev-key")

    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "14"))
    min_history_days: int = int(os.getenv("MIN_HISTORY_DAYS", "28"))
    backtest_days: int = int(os.getenv("BACKTEST_DAYS", "14"))
    # candidates are tried in a backtest; best WAPE wins. Missing libs are skipped.
    model_candidates: tuple = tuple(
        os.getenv("MODEL_CANDIDATES", "baseline,prophet,lstm").split(",")
    )

    # scheduler (cron syntax, server local time)
    enable_scheduler: bool = _bool("ENABLE_SCHEDULER", "true")
    batch_cron: str = os.getenv("BATCH_CRON", "0 5 * * *")      # daily scoring
    retrain_cron: str = os.getenv("RETRAIN_CRON", "0 2 * * 0")  # weekly retrain (Sun 02:00)

    # risk-score weights (see app/scoring.py)
    unsold_weight: float = float(os.getenv("W_UNSOLD", "0.6"))
    cover_weight: float = float(os.getenv("W_COVER", "0.4"))
    uncertainty_weight: float = float(os.getenv("W_PESSIMISTIC", "0.35"))


settings = Settings()

