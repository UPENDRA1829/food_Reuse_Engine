"""
Database layer.

>>> ADAPTER POINT: if your Week 1-2 tables use different names/columns, change ONLY this
file (rename __tablename__/columns or point these models at SQL views). Every other
module talks to these ORM classes.  All tables carry tenant_id (multi-tenant).
"""
from contextlib import contextmanager
from datetime import date, datetime

from sqlalchemy import (Date, DateTime, Float, Index, Integer, String, Text,
                        UniqueConstraint, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


# ---------- source data (owned by Weeks 1-2) ----------
class SaleRecord(Base):
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    sale_date: Mapped[date] = mapped_column(Date)
    quantity: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_sales_t_c_d", "tenant_id", "category", "sale_date"),)


class PurchaseRecord(Base):
    __tablename__ = "purchases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    purchase_date: Mapped[date] = mapped_column(Date)
    quantity: Mapped[float] = mapped_column(Float)
    unit_cost: Mapped[float] = mapped_column(Float, default=0.0)
    __table_args__ = (Index("ix_purch_t_c_d", "tenant_id", "category", "purchase_date"),)


class WasteRecord(Base):
    __tablename__ = "waste_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    waste_date: Mapped[date] = mapped_column(Date)
    quantity: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(32), default="expired")
    __table_args__ = (Index("ix_waste_t_c_d", "tenant_id", "category", "waste_date"),)


class InventoryItem(Base):
    """One row per product batch currently in stock."""
    __tablename__ = "inventory_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    batch_id: Mapped[str] = mapped_column(String(64), default="default")
    name: Mapped[str] = mapped_column(String(128), default="")
    category: Mapped[str] = mapped_column(String(64))
    storage_type: Mapped[str] = mapped_column(String(32), default="ambient")  # ambient|chilled|frozen
    perishability: Mapped[str] = mapped_column(String(16), default="medium")  # high|medium|low
    current_stock: Mapped[float] = mapped_column(Float)
    expiry_date: Mapped[date] = mapped_column(Date)
    shelf_life_days: Mapped[int] = mapped_column(Integer, default=7)
    unit_cost: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    lead_time_days: Mapped[int] = mapped_column(Integer, default=2)
    pack_size: Mapped[int] = mapped_column(Integer, default=1)
    min_order_qty: Mapped[int] = mapped_column(Integer, default=1)
    on_order_qty: Mapped[float] = mapped_column(Float, default=0.0)
    __table_args__ = (UniqueConstraint("tenant_id", "product_id", "batch_id"),)


class StorageCapacity(Base):
    __tablename__ = "storage_capacity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    storage_type: Mapped[str] = mapped_column(String(32))
    capacity_units: Mapped[float] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("tenant_id", "storage_type"),)


# ---------- outputs (owned by Weeks 3-4) ----------
class ForecastRecord(Base):
    __tablename__ = "demand_forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    forecast_date: Mapped[date] = mapped_column(Date)
    yhat: Mapped[float] = mapped_column(Float)
    yhat_lower: Mapped[float] = mapped_column(Float)
    yhat_upper: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(32))
    generated_at: Mapped[datetime] = mapped_column(DateTime)
    __table_args__ = (Index("ix_fc_t_c_d", "tenant_id", "category", "forecast_date"),)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(32))
    wape: Mapped[float] = mapped_column(Float)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    model_path: Mapped[str] = mapped_column(String(256), default="")
    trained_at: Mapped[datetime] = mapped_column(DateTime)


class RiskPrediction(Base):
    __tablename__ = "waste_risk_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    batch_id: Mapped[str] = mapped_column(String(64))
    scored_at: Mapped[datetime] = mapped_column(DateTime)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    expected_waste_units: Mapped[float] = mapped_column(Float)
    expected_waste_value: Mapped[float] = mapped_column(Float)
    days_to_expiry: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(48))
    __table_args__ = (Index("ix_risk_t_s", "tenant_id", "scored_at"),)


class ReorderRecommendation(Base):
    __tablename__ = "reorder_recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    generated_at: Mapped[datetime] = mapped_column(DateTime)
    recommended_qty: Mapped[float] = mapped_column(Float)
    order_by_date: Mapped[date] = mapped_column(Date)
    reasons: Mapped[str] = mapped_column(Text, default="")
    __table_args__ = (Index("ix_reorder_t_g", "tenant_id", "generated_at"),)


_engine = None
_Session = None


def get_engine():
    global _engine, _Session
    if _engine is None:
        kwargs = {"connect_args": {"check_same_thread": False}} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, future=True, **kwargs)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope():
    get_engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
