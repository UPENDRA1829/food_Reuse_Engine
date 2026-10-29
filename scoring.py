
"""
Waste-risk scoring.

For every inventory batch, simulate FEFO (first-expiry-first-out)
sell-down against the supplied demand estimates.

Important:
- Risk scores are operational estimates, not proof of food safety.
- Expired stock must be isolated and assessed.
- This module does not determine whether a forecast is reliable.
- Unknown unit costs must not be reported as a confirmed zero waste value.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np

from .config import settings

TAU = {"high": 3.0, "medium": 7.0, "low": 21.0}
PERISH_W = {"high": 1.0, "medium": 0.6, "low": 0.25}
LEVELS = [(25, "LOW"), (50, "MEDIUM"), (75, "HIGH"), (101, "CRITICAL")]

PRELIMINARY_WARNING = (
    "Preliminary risk estimate. Forecast reliability must be checked "
    "separately using observed historical data."
)


@dataclass
class Batch:
    product_id: str
    batch_id: str
    stock: float
    expiry_date: date
    perishability: str = "medium"
    unit_cost: float | None = None
    name: str = ""
    category: str = ""


def level_for(score: float) -> str:
    for cap, name in LEVELS:
        if score < cap:
            return name
    return "CRITICAL"


def extend_demand(daily: np.ndarray, days: int) -> np.ndarray:
    """Extend the supplied demand series by repeating its weekly pattern."""
    daily = np.asarray(daily, dtype=float)

    if days <= 0:
        return np.zeros(0, dtype=float)

    if len(daily) >= days:
        return daily[:days]

    if len(daily) == 0:
        return np.zeros(days, dtype=float)

    pattern = daily[-7:] if len(daily) >= 7 else np.resize(daily, 7)
    reps = int(np.ceil((days - len(daily)) / 7))

    return np.concatenate([daily, np.tile(pattern, reps)])[:days]


def simulate_fefo(
    batches: list[Batch],
    demand: np.ndarray,
    today: date,
) -> dict[str, float]:
    """
    Simulate daily sell-down using FEFO.

    demand[0] is today's demand; demand[t] is demand t days out.
    A batch with d days to expiry can sell on days 0..d-1.
    Already-expired stock is not sold by this simulation.

    Returns:
        Mapping of batch_id to estimated units left at expiry.
    """
    order = sorted(batches, key=lambda b: b.expiry_date)

    remaining = {
        b.batch_id: max(0.0, b.stock)
        for b in order
    }

    horizon = max(
        [(b.expiry_date - today).days for b in order] + [0]
    )
    dem = extend_demand(demand, horizon + 1)

    leftover: dict[str, float] = {}

    for t in range(horizon + 1):
        # Mark batches expired by this day before selling today's demand.
        for b in order:
            if b.batch_id in leftover:
                continue

            if (b.expiry_date - today).days <= t:
                leftover[b.batch_id] = remaining[b.batch_id]

        need = max(0.0, float(dem[t]))

        for b in order:
            if b.batch_id in leftover or need <= 0:
                continue

            take = min(remaining[b.batch_id], need)
            remaining[b.batch_id] -= take
            need -= take

    # Preserve all batches in the result, including already-expired batches.
    for b in order:
        leftover.setdefault(b.batch_id, remaining[b.batch_id])

    return leftover


def _action(
    level: str,
    dte: int,
    expected_waste: float,
    expired: bool,
) -> str:
    """Choose a stock-handling action; this is not a food-safety decision."""
    if expired:
        return "ISOLATE_AND_ASSESS_SAFETY"

    if level == "CRITICAL":
        return "DONATE_OR_MARKDOWN_NOW" if dte <= 2 else "MARKDOWN_AND_PROMOTE"

    if level == "HIGH":
        return "MARKDOWN_OR_BUNDLE" if dte <= 5 else "PROMOTE_AND_PAUSE_REORDER"

    if level == "MEDIUM":
        return "PAUSE_REORDER_AND_MONITOR"

    return "NO_ACTION"


def score_product(
    batches: list[Batch],
    demand_mean,
    demand_low,
    today: date | None = None,
    category_waste_rate: float = 0.0,
) -> list[dict]:
    """
    Score every batch of one product using supplied mean and pessimistic
    demand estimates.

    This function does not verify whether those estimates came from a
    sufficiently trained or reliable forecasting model.
    """
    today = today or date.today()

    demand_mean = np.asarray(demand_mean, dtype=float)
    demand_low = np.asarray(demand_low, dtype=float)

    # Avoid invalid negative demand estimates in sell-down calculations.
    demand_mean = np.maximum(demand_mean, 0.0)
    demand_low = np.maximum(demand_low, 0.0)

    base = simulate_fefo(batches, demand_mean, today)
    pess = simulate_fefo(batches, demand_low, today)

    avg_daily = (
        float(demand_mean[:14].mean())
        if len(demand_mean)
        else 0.0
    )

    total_stock = sum(max(0.0, b.stock) for b in batches)
    uw = min(1.0, max(0.0, settings.uncertainty_weight))

    results = []

    for b in batches:
        dte = (b.expiry_date - today).days
        expired = dte <= 0

        stock = max(0.0, b.stock)

        if stock <= 0:
            unsold = 0.0
            ratio = 0.0
        else:
            ratio_base = min(1.0, max(0.0, base[b.batch_id] / stock))
            ratio_pess = min(1.0, max(0.0, pess[b.batch_id] / stock))

            ratio = (1.0 - uw) * ratio_base + uw * ratio_pess
            unsold = min(stock, max(0.0, base[b.batch_id]))

        cover_days = (
            total_stock / avg_daily
            if avg_daily > 1e-9
            else 1e9
        )

        cover = min(
            1.0,
            max(0.0, (cover_days / max(dte, 1)) / 2.0),
        )

        core = (
            settings.unsold_weight * ratio
            + settings.cover_weight * cover
        )

        perish = b.perishability if b.perishability in TAU else "medium"

        urgency = float(
            np.exp(-max(dte, 0) / TAU[perish])
        )

        waste_prior = min(
            1.0,
            max(0.0, category_waste_rate / 0.2),
        )

        modulation = (
            0.60
            + 0.25 * urgency
            + 0.10 * PERISH_W[perish]
            + 0.05 * waste_prior
        )

        score = (
            100.0
            if expired and stock > 0
            else 100.0 * core * modulation
        )

        score = float(round(min(100.0, max(0.0, score)), 1))
        level = level_for(score)

        # None means the cost is unknown, not that the waste has no value.
        valid_cost = (
            b.unit_cost is not None
            and np.isfinite(b.unit_cost)
            and b.unit_cost >= 0
        )

        waste_value = (
            round(unsold * b.unit_cost, 2)
            if valid_cost
            else None
        )

        results.append({
            "product_id": b.product_id,
            "batch_id": b.batch_id,
            "name": b.name,
            "category": b.category,
            "stock": b.stock,
            "days_to_expiry": dte,
            "risk_score": score,
            "risk_level": level,
            "expected_unsold_units": round(unsold, 1),
            "expected_waste_value": waste_value,
            "unit_cost_known": valid_cost,
            "days_of_cover": (
                None if cover_days >= 1e8 else round(cover_days, 1)
            ),
            "prediction_status": "preliminary",
            "warning": PRELIMINARY_WARNING,
            "action": _action(
                level,
                dte,
                unsold,
                expired and stock > 0,
            ),
        })

    return results


def to_dicts(rows):
    """Convert dataclass instances to dictionaries where applicable."""
    return [
        asdict(r) if hasattr(r, "__dataclass_fields__") else r
        for r in rows
    ]
