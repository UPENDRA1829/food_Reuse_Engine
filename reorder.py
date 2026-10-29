"""
Smart reorder recommendations.

For each product:
  target stock   = forecast demand over the review window after arrival + safety stock
  safety stock   = z * sigma_daily * sqrt(lead + review)       (z lower for perishables)
  shelf-life cap = never order more than can be sold before the NEW batch expires
  usable on hand = stock that the sell-down simulation says will actually sell (not wasted)
  order          = target - (usable on hand at arrival + on order)  -> MOQ / pack rounding
then orders are squeezed into the remaining storage capacity per storage type, most urgent
(lowest days of cover) first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import math

import numpy as np

from .scoring import extend_demand

Z = {"high": 1.04, "medium": 1.28, "low": 1.65}  # ~85% / 90% / 95% service level


@dataclass
class ProductState:
    product_id: str
    storage_type: str = "ambient"
    perishability: str = "medium"
    stock_on_hand: float = 0.0
    sellable_stock: float = 0.0       # stock expected to sell before expiry
    on_order: float = 0.0
    lead_time_days: int = 2
    review_period_days: int = 7
    shelf_life_days: int = 7
    min_order_qty: int = 1
    pack_size: int = 1
    unit_cost: float = 0.0
    unit_volume: float = 1.0          # storage units consumed per item


@dataclass
class Recommendation:
    product_id: str
    recommended_qty: float
    order_by_date: date
    target_stock: float
    safety_stock: float
    order_cost: float
    days_of_cover: float | None
    reasons: list[str] = field(default_factory=list)


def _round_pack(q: float, pack: int, moq: int, cap: float) -> float:
    pack = max(1, pack)
    qty = math.ceil(q / pack) * pack
    qty = max(qty, moq)
    qty = math.ceil(qty / pack) * pack
    if qty > cap:                       # rounding up would breach the cap -> round down
        qty = math.floor(cap / pack) * pack
        if qty < moq:
            return 0.0
    return float(qty)


def recommend(products: list[ProductState], demand_mean: dict[str, np.ndarray],
              sigma_daily: dict[str, float], capacity: dict[str, float] | None = None,
              used: dict[str, float] | None = None, today: date | None = None) -> list[Recommendation]:
    today = today or date.today()
    recs: list[Recommendation] = []
    for p in products:
        L, R = max(0, p.lead_time_days), max(1, p.review_period_days)
        horizon = max(L + R, p.shelf_life_days) + 2
        d = extend_demand(demand_mean.get(p.product_id, np.zeros(1)), horizon)
        avg = float(d[:7].mean())
        reasons: list[str] = []

        sigma = sigma_daily.get(p.product_id, 0.0)
        safety = Z.get(p.perishability, 1.28) * sigma * math.sqrt(L + R)
        window_demand = float(d[L:L + R].sum())
        target = window_demand + safety

        arrive_stock = max(0.0, p.sellable_stock - float(d[:L].sum())) + p.on_order
        raw = max(0.0, target - arrive_stock)

        # new batch must sell before it expires (keep 1 day of margin)
        sell_window = max(1, p.shelf_life_days - 1)
        shelf_cap = float(d[L:L + sell_window].sum()) - arrive_stock
        shelf_cap = max(0.0, shelf_cap)
        if raw > shelf_cap:
            reasons.append("capped_by_shelf_life")
            raw = shelf_cap

        qty = _round_pack(raw, p.pack_size, p.min_order_qty, cap=shelf_cap) if raw > 0 else 0.0
        if raw > 0 and qty == 0:
            reasons.append("below_pack_or_min_order_size")

        cum = np.cumsum(d)
        stockout_day = int(np.argmax(cum >= p.sellable_stock)) if cum[-1] >= p.sellable_stock > 0 else (0 if p.sellable_stock <= 0 else horizon)
        order_by = today + timedelta(days=max(0, stockout_day - L))
        cover = (p.stock_on_hand / avg) if avg > 1e-9 else None
        if p.sellable_stock < p.stock_on_hand * 0.9:
            reasons.append("existing_stock_will_partly_expire_so_not_counted")
        if qty == 0 and not reasons:
            reasons.append("sufficient_stock")
        recs.append(Recommendation(p.product_id, qty, order_by, round(target, 1), round(safety, 1),
                                   round(qty * p.unit_cost, 2), None if cover is None else round(cover, 1), reasons))

    # ---- storage capacity (greedy, most urgent first)
    if capacity:
        used = dict(used or {})
        by_pid = {p.product_id: p for p in products}
        for rec in sorted(recs, key=lambda r: (r.days_of_cover is None, r.days_of_cover or 0)):
            p = by_pid[rec.product_id]
            if rec.recommended_qty <= 0 or p.storage_type not in capacity:
                continue
            free = capacity[p.storage_type] - used.get(p.storage_type, 0.0)
            need = rec.recommended_qty * p.unit_volume
            if need > free:
                fit = max(0.0, free) / p.unit_volume
                new_qty = math.floor(fit / max(1, p.pack_size)) * max(1, p.pack_size)
                if new_qty < p.min_order_qty:
                    new_qty = 0
                rec.reasons.append("reduced_by_storage_capacity" if new_qty else "blocked_by_storage_capacity")
                rec.recommended_qty = float(new_qty)
                rec.order_cost = round(new_qty * p.unit_cost, 2)
                need = new_qty * p.unit_volume
            used[p.storage_type] = used.get(p.storage_type, 0.0) + need
    return recs
