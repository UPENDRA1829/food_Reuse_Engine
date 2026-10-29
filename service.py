"""Orchestration: train -> forecast -> score -> reorder, per tenant."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import settings
from .data.preprocess import (
    build_category_dataset,
    category_waste_rate,
    count_observed_sales_days,
    list_categories,
    list_tenants,
    product_shares,
)
from .db import (ForecastRecord, InventoryItem, ModelRun, ReorderRecommendation, RiskPrediction,
                 StorageCapacity, session_scope)
from .models.forecast import (REGISTRY, BaselineForecaster, InsufficientData, select_and_fit)
from .reorder import ProductState, recommend
from .scoring import Batch, score_product, simulate_fefo

log = logging.getLogger(__name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _model_path(tenant: str, category: str) -> str:
    os.makedirs(os.path.join(settings.model_dir, tenant), exist_ok=True)
    return os.path.join(settings.model_dir, tenant, f"{category}.joblib")


# ------------------------------------------------------------------ training & forecasting
def train_category(session: Session, tenant: str, category: str) -> dict | None:
    ds = build_category_dataset(session, tenant, category)
    if ds.empty or len(ds) < settings.min_history_days:
        log.info("skip %s/%s: only %d days of history", tenant, category, len(ds))
        return None
    sel = select_and_fit(ds["y"], settings.model_candidates, settings.backtest_days)
    path = _model_path(tenant, category)
    joblib.dump(sel.model, path)
    now = _utcnow()
    session.add(ModelRun(tenant_id=tenant, category=category, model_name=sel.name,
                         wape=float(sel.wape) if not np.isnan(sel.wape) else -1.0,
                         metrics_json=json.dumps(sel.scores), model_path=path, trained_at=now))
    session.commit()
    _write_forecast(session, tenant, category, sel.model, sel.name, ds["y"])
    return {"tenant_id": tenant, "category": category, "model": sel.name,
            "wape": None if np.isnan(sel.wape) else round(sel.wape, 4),
            "candidate_wape": {k: (None if np.isnan(v) else round(v, 4)) for k, v in sel.scores.items()},
            "history_days": len(ds)}


def _write_forecast(session, tenant, category, model, name, history: pd.Series):
    start = date.today()
    fc = model.predict(start, settings.forecast_horizon, history=history)
    session.execute(delete(ForecastRecord).where(
        ForecastRecord.tenant_id == tenant, ForecastRecord.category == category))
    now = _utcnow()
    for ds, r in fc.iterrows():
        session.add(ForecastRecord(tenant_id=tenant, category=category, forecast_date=ds.date(),
                                   yhat=float(r.yhat), yhat_lower=float(r.yhat_lower),
                                   yhat_upper=float(r.yhat_upper), model_name=name, generated_at=now))
    session.commit()


def train_tenant(tenant: str, category: str | None = None) -> list[dict]:
    out = []
    with session_scope() as s:
        for cat in ([category] if category else list_categories(s, tenant)):
            r = train_category(s, tenant, cat)
            if r:
                out.append(r)
    return out


def refresh_forecasts(session: Session, tenant: str) -> int:
    """Refresh forecasts only when sufficient real sales history exists."""
    n = 0

    for cat in list_categories(session, tenant):
        ds = build_category_dataset(session, tenant, cat)

        observed_days = count_observed_sales_days(
            session, tenant, cat
        )

        if ds.empty or observed_days < settings.min_history_days:
            log.info(
                "Skipping forecast refresh for %s/%s: "
                "%d observed sales days; %d required",
                tenant,
                cat,
                observed_days,
                settings.min_history_days,
            )
            continue

        path = _model_path(tenant, cat)

        try:
            model = joblib.load(path)
        except Exception:
            model = BaselineForecaster().fit(ds["y"])

        _write_forecast(
            session, tenant, cat, model, model.name, ds["y"]
        )

        n += 1

    return n

def get_forecast(
    session: Session,
    tenant: str,
    category: str,
    horizon: int | None = None
) -> pd.DataFrame:
    """Return forecasts only when sufficient observed sales history exists."""

    # Step 1: Check actual recorded sales days.
    observed_days = count_observed_sales_days(
        session, tenant, category
    )

    if observed_days < settings.min_history_days:
        raise InsufficientData(
            f"Insufficient sales history for {tenant}/{category}: "
            f"{observed_days} observed days; "
            f"{settings.min_history_days} required"
        )

    # Step 2: Check whether a cached forecast is available.
    horizon = horizon or settings.forecast_horizon

    rows = session.execute(
        select(ForecastRecord).where(
            ForecastRecord.tenant_id == tenant,
            ForecastRecord.category == category,
            ForecastRecord.forecast_date >= date.today()
        ).order_by(ForecastRecord.forecast_date)
    ).scalars().all()

    if len(rows) >= min(horizon, 7):
        df = pd.DataFrame([
            {
                "ds": r.forecast_date,
                "yhat": r.yhat,
                "yhat_lower": r.yhat_lower,
                "yhat_upper": r.yhat_upper,
                "model": r.model_name
            }
            for r in rows
        ]).set_index("ds")

        return df.iloc[:horizon]

    # Step 3: Build a new forecast if no suitable cache exists.
    ds = build_category_dataset(session, tenant, category)

    if ds.empty:
        raise InsufficientData(
            f"No usable sales history for {tenant}/{category}"
        )

    model = BaselineForecaster().fit(ds["y"])

    df = model.predict(date.today(), horizon)
    df["model"] = "baseline_live"
    df.index = [d.date() for d in df.index]

    return df

# ------------------------------------------------------------------ scoring & reorder
def _product_demand(session, tenant, category, pid, horizon, cache):
    """Return (mean, pessimistic, sigma_daily) arrays for a product from its category forecast."""
    if category not in cache:
        fc = get_forecast(session, tenant, category, horizon)
        cache[category] = (fc, product_shares(session, tenant, category),
                           category_waste_rate(session, tenant, category))
    fc, shares, _ = cache[category]
    share = shares.get(pid, 0.0)
    mean = fc["yhat"].values * share
    low = fc["yhat_lower"].values * share
    sigma = float(((fc["yhat_upper"] - fc["yhat_lower"]) / (2 * 1.2816)).mean()) * share
    return mean, low, sigma


def _batches(items) -> dict[str, list[InventoryItem]]:
    g: dict[str, list[InventoryItem]] = {}
    for it in items:
        g.setdefault(it.product_id, []).append(it)
    return g


def score_items(session: Session, tenant: str, items: list[InventoryItem] | None = None,
                override_demand: dict[str, float] | None = None, persist: bool = False) -> list[dict]:
    if items is None:
        items = session.execute(select(InventoryItem).where(InventoryItem.tenant_id == tenant)).scalars().all()
    cache: dict = {}
    results: list[dict] = []
    H = max(settings.forecast_horizon, 14)
    for pid, rows in _batches(items).items():
        cat = rows[0].category
        try:
            mean, low, _ = _product_demand(session, tenant, cat, pid, H, cache)
            src = "forecast"
        except InsufficientData:
            mean = low = np.zeros(H)
            src = "no_history"
        if override_demand and pid in override_demand:
            mean = low = np.full(H, float(override_demand[pid]))
            src = "override"
        waste_rate = cache.get(cat, (None, None, 0.0))[2]
        batches = [Batch(r.product_id, r.batch_id, r.current_stock, r.expiry_date, r.perishability,
                         r.unit_cost, r.name, r.category) for r in rows]
        for res in score_product(batches, mean, low, category_waste_rate=waste_rate):
            res["demand_source"] = src
            results.append(res)
    results.sort(key=lambda r: -r["risk_score"])
    if persist:
        now = _utcnow()
        for r in results:
            session.add(RiskPrediction(
                tenant_id=tenant, product_id=r["product_id"], batch_id=r["batch_id"], scored_at=now,
                risk_score=r["risk_score"], risk_level=r["risk_level"],
                expected_waste_units=r["expected_unsold_units"], expected_waste_value=r["expected_waste_value"],
                days_to_expiry=r["days_to_expiry"], action=r["action"]))
        session.commit()
    return results


def recommend_reorders(session: Session, tenant: str, persist: bool = False,
                       review_period_days: int = 7) -> list[dict]:
    items = session.execute(select(InventoryItem).where(InventoryItem.tenant_id == tenant)).scalars().all()
    caps = {c.storage_type: c.capacity_units for c in session.execute(
        select(StorageCapacity).where(StorageCapacity.tenant_id == tenant)).scalars()}
    cache: dict = {}
    H = max(settings.forecast_horizon, 14)
    states, demand, sigma = [], {}, {}
    used: dict[str, float] = {}
    for pid, rows in _batches(items).items():
        r0 = rows[0]
        try:
            mean, low, sg = _product_demand(session, tenant, r0.category, pid, H, cache)
        except InsufficientData:
            continue
        today = date.today()
        batches = [Batch(r.product_id, r.batch_id, r.current_stock, r.expiry_date) for r in rows]
        left = simulate_fefo(batches, mean, today)
        stock = sum(r.current_stock for r in rows)
        sellable = max(0.0, stock - sum(left.values()))
        states.append(ProductState(
            product_id=pid, storage_type=r0.storage_type, perishability=r0.perishability,
            stock_on_hand=stock, sellable_stock=sellable, on_order=sum(r.on_order_qty for r in rows),
            lead_time_days=r0.lead_time_days, review_period_days=review_period_days,
            shelf_life_days=r0.shelf_life_days, min_order_qty=r0.min_order_qty,
            pack_size=r0.pack_size, unit_cost=r0.unit_cost))
        demand[pid], sigma[pid] = mean, sg
        used[r0.storage_type] = used.get(r0.storage_type, 0.0) + stock
    recs = recommend(states, demand, sigma, capacity=caps, used=used)
    out = [{"product_id": r.product_id, "recommended_qty": r.recommended_qty,
            "order_by_date": r.order_by_date.isoformat(), "target_stock": r.target_stock,
            "safety_stock": r.safety_stock, "estimated_cost": r.order_cost,
            "days_of_cover": r.days_of_cover, "reasons": r.reasons} for r in recs]
    out.sort(key=lambda r: (r["order_by_date"], -r["recommended_qty"]))
    if persist:
        now = _utcnow()
        for r in out:
            session.add(ReorderRecommendation(
                tenant_id=tenant, product_id=r["product_id"], generated_at=now,
                recommended_qty=r["recommended_qty"], order_by_date=date.fromisoformat(r["order_by_date"]),
                reasons=",".join(r["reasons"])))
        session.commit()
    return out


# ------------------------------------------------------------------ batch jobs
BATCH_STATUS: dict = {"last_batch": None, "last_retrain": None}


def run_daily_batch() -> dict:
    summary = {"started": _utcnow().isoformat(), "tenants": {}}
    with session_scope() as s:
        for t in list_tenants(s):
            try:
                n = refresh_forecasts(s, t)
                scored = score_items(s, t, persist=True)
                recs = recommend_reorders(s, t, persist=True)
                summary["tenants"][t] = {"forecasts": n, "scored": len(scored),
                                         "critical": sum(r["risk_level"] == "CRITICAL" for r in scored),
                                         "reorders": sum(r["recommended_qty"] > 0 for r in recs)}
            except Exception as e:  # one bad tenant must not stop the rest
                log.exception("batch failed for %s", t)
                summary["tenants"][t] = {"error": str(e)}
    summary["finished"] = _utcnow().isoformat()
    BATCH_STATUS["last_batch"] = summary
    return summary


def run_retrain() -> dict:
    summary = {"started": _utcnow().isoformat(), "tenants": {}}
    with session_scope() as s:
        tenants = list_tenants(s)
    for t in tenants:
        try:
            summary["tenants"][t] = train_tenant(t)
        except Exception as e:
            log.exception("retrain failed for %s", t)
            summary["tenants"][t] = {"error": str(e)}
    summary["finished"] = _utcnow().isoformat()
    BATCH_STATUS["last_retrain"] = summary
    return summary
