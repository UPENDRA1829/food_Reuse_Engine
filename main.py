"""FastAPI microservice: scheduled batch predictions + real-time on-demand scoring."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import service
from .foodrescue_adapter import get_summary, get_inventory, get_sales
from .foodrescue_risk_adapter import foodrescue_inventory_to_items
from .config import settings
from .data.preprocess import export_datasets
from .db import InventoryItem, RiskPrediction, ReorderRecommendation, init_db, session_scope
from .models.forecast import InsufficientData
from .scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.enable_scheduler:
        start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Waste Prediction Service", version="1.0.0", lifespan=lifespan)


def auth(x_api_key: str = Header(default="")):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "invalid API key")


# ---------------------------------------------------------------- schemas
class ItemIn(BaseModel):
    product_id: str
    batch_id: str = "default"
    name: str = ""
    category: str
    current_stock: float = Field(ge=0)
    expiry_date: date
    perishability: str = "medium"
    unit_cost: float = 0.0
    shelf_life_days: int = 7
    daily_demand_override: float | None = Field(default=None, ge=0, description="optional manual demand/day")


class ScoreRequest(BaseModel):
    items: list[ItemIn] = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "time": _utcnow().isoformat(),
            "candidates": settings.model_candidates, **service.BATCH_STATUS}


@app.post("/v1/{tenant_id}/train", dependencies=[Depends(auth)])
def train(tenant_id: str, bg: BackgroundTasks, category: str | None = None, sync: bool = False):
    """Fit/select models (Baseline/Prophet/LSTM backtest). Async by default; sync=true waits."""
    if sync:
        return {"trained": service.train_tenant(tenant_id, category)}
    bg.add_task(service.train_tenant, tenant_id, category)
    return {"status": "training_started"}


@app.get("/v1/{tenant_id}/forecast/{category}", dependencies=[Depends(auth)])
def forecast(tenant_id: str, category: str, horizon: int = Query(14, ge=1, le=60)):
    with session_scope() as s:
        try:
            df = service.get_forecast(s, tenant_id, category, horizon)
        except InsufficientData as e:
            raise HTTPException(404, str(e))
    daily = [{"date": str(i), **{k: round(float(v), 2) if k != "model" else v for k, v in r.items()}}
             for i, r in df.iterrows()]
    return {"category": category, "daily": daily,
            "weekly_total": round(float(df["yhat"].iloc[:7].sum()), 1),
            "model": df["model"].iloc[0]}


@app.post("/v1/{tenant_id}/risk/score", dependencies=[Depends(auth)])
def score_on_demand(tenant_id: str, req: ScoreRequest):
    """Real-time scoring of items supplied in the request (e.g. straight from the POS)."""
    items = [InventoryItem(tenant_id=tenant_id, product_id=i.product_id, batch_id=i.batch_id, name=i.name,
                           category=i.category, current_stock=i.current_stock, expiry_date=i.expiry_date,
                           perishability=i.perishability, unit_cost=i.unit_cost,
                           shelf_life_days=i.shelf_life_days) for i in req.items]
    override = {i.product_id: i.daily_demand_override for i in req.items if i.daily_demand_override is not None}
    with session_scope() as s:
        return {"results": service.score_items(s, tenant_id, items, override)}


@app.get("/v1/{tenant_id}/risk", dependencies=[Depends(auth)])
def risk_live(tenant_id: str, min_level: str = "LOW", limit: int = Query(100, le=1000)):
    """Score the tenant's current DB inventory right now."""
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if min_level not in order:
        raise HTTPException(422, f"min_level must be one of {order}")
    with session_scope() as s:
        res = service.score_items(s, tenant_id)
    res = [r for r in res if order.index(r["risk_level"]) >= order.index(min_level)][:limit]
    return {"count": len(res), "results": res}


@app.get("/v1/{tenant_id}/risk/latest", dependencies=[Depends(auth)])
def risk_latest_batch(tenant_id: str, limit: int = Query(100, le=1000)):
    """Results of the last scheduled batch run (cheap read, no computation)."""
    with session_scope() as s:
        last = s.execute(select(RiskPrediction.scored_at).where(RiskPrediction.tenant_id == tenant_id)
                         .order_by(RiskPrediction.scored_at.desc()).limit(1)).scalar()
        if not last:
            raise HTTPException(404, "no batch run yet; POST /v1/batch/run")
        rows = s.execute(select(RiskPrediction).where(
            RiskPrediction.tenant_id == tenant_id, RiskPrediction.scored_at == last)
            .order_by(RiskPrediction.risk_score.desc()).limit(limit)).scalars().all()
        return {"scored_at": last.isoformat(), "results": [
            {c: getattr(r, c) for c in ("product_id", "batch_id", "risk_score", "risk_level",
                                        "expected_waste_units", "expected_waste_value",
                                        "days_to_expiry", "action")} for r in rows]}


@app.get("/v1/{tenant_id}/reorder", dependencies=[Depends(auth)])
def reorder(tenant_id: str, review_period_days: int = Query(7, ge=1, le=30), only_orders: bool = True):
    with session_scope() as s:
        recs = service.recommend_reorders(s, tenant_id, review_period_days=review_period_days)
    if only_orders:
        recs = [r for r in recs if r["recommended_qty"] > 0]
    return {"count": len(recs), "total_estimated_cost": round(sum(r["estimated_cost"] for r in recs), 2),
            "recommendations": recs}


@app.post("/v1/batch/run", dependencies=[Depends(auth)])
def run_batch(bg: BackgroundTasks, retrain: bool = False, sync: bool = False):
    fn = service.run_retrain if retrain else service.run_daily_batch
    if sync:
        return fn()
    bg.add_task(fn)
    return {"status": "started", "job": "retrain" if retrain else "daily_batch"}


@app.post("/v1/{tenant_id}/datasets/export", dependencies=[Depends(auth)])
def export(tenant_id: str):
    with session_scope() as s:
        return {"files": export_datasets(s, tenant_id, f"./datasets/{tenant_id}")}
# ----------------------------------------------------------------
# FoodRescue integration endpoints (read-only)
# ----------------------------------------------------------------

@app.get("/v1/foodrescue/summary", dependencies=[Depends(auth)])
def foodrescue_summary():
    """Return FoodRescue data counts and machine-learning readiness."""
    try:
        return get_summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Failed to read FoodRescue summary")
        raise HTTPException(
            status_code=500,
            detail="Unable to read FoodRescue summary."
        )


@app.get("/v1/foodrescue/inventory", dependencies=[Depends(auth)])
def foodrescue_inventory(
    limit: int = Query(100, ge=1, le=1000)
):
    """Read FoodRescue inventory without modifying its database."""
    try:
        items = get_inventory(limit)
        return {"count": len(items), "results": items}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Failed to read FoodRescue inventory")
        raise HTTPException(
            status_code=500,
            detail="Unable to read FoodRescue inventory."
        )


@app.get("/v1/foodrescue/sales", dependencies=[Depends(auth)])
def foodrescue_sales(
    limit: int = Query(100, ge=1, le=1000)
):
    """Read FoodRescue sales without modifying its database."""
    try:
        records = get_sales(limit)
        return {"count": len(records), "results": records}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Failed to read FoodRescue sales")
        raise HTTPException(
            status_code=500,
            detail="Unable to read FoodRescue sales."
        )
@app.get("/v1/foodrescue/risk-preview", dependencies=[Depends(auth)])
def foodrescue_risk_preview(
    limit: int = Query(100, ge=1, le=1000)
):
    """Preview preliminary FoodRescue inventory risk scores without saving them."""
    try:
        records = get_inventory(limit)
        items = foodrescue_inventory_to_items(records)

        if not items:
            return {
                "count": 0,
                "results": [],
                "warning": "No inventory records with valid expiry dates were found."
            }

        results = []
        warnings = []

        with session_scope() as session:
            for item in items:
                try:
                    scored = service.score_items(
                        session,
                        item.tenant_id,
                        items=[item],
                        persist=False
                    )
                    for result in scored:
                        result["source"] = "FoodRescue"
                        result["prediction_status"] = "preliminary"
                        result["warning"] = (
                            "Sales and waste history are insufficient for "
                            "a reliable AI demand forecast."
                        )
                        results.append(result)
                except Exception:
                    logging.exception(
                        "Could not score FoodRescue inventory %s",
                        item.product_id
                    )
                    warnings.append(item.product_id)

        return {
            "count": len(results),
            "results": results,
            "failed_inventory_items": warnings,
            "saved_to_database": False,
            "warning": (
                "Preliminary risk estimates only. FoodRescue has limited "
                "sales history and no confirmed waste history."
            )
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("FoodRescue risk preview failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate FoodRescue risk preview."
        )

@app.get("/v1/foodrescue/sales-readiness", dependencies=[Depends(auth)])
def foodrescue_sales_readiness():
    """Report observed FoodRescue sales history without training a model."""
    from .foodrescue_adapter import get_daily_sales_history

    try:
        records = get_daily_sales_history(10000)
        categories = sorted({
            str(row["category"]).strip()
            for row in records
            if row.get("category")
        })

        results = []
        for category in categories:
            rows = [
                row for row in records
                if str(row.get("category", "")).strip() == category
            ]
            observed_dates = {
                str(row["sale_date"])[:10]
                for row in rows
                if row.get("sale_date")
            }
            total_units = sum(
                float(row.get("quantity") or 0)
                for row in rows
            )

            results.append({
                "category": category,
                "observed_sales_days": len(observed_dates),
                "total_units_sold": round(total_units, 2),
                "forecast_ready": len(observed_dates) >= settings.min_history_days,
                "required_observed_days": settings.min_history_days,
            })

        return {
            "source": "FoodRescue",
            "database_mode": "read-only",
            "count": len(results),
            "results": results,
            "model_training_performed": False,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("FoodRescue sales-readiness check failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to assess FoodRescue sales-history readiness."
        )
