
"""Read-only bridge from FoodRescue inventory to waste-risk scoring."""

from datetime import date
from typing import Any

from .db import InventoryItem


def foodrescue_inventory_to_items(
    records: list[dict[str, Any]],
) -> list[InventoryItem]:
    """Convert FoodRescue inventory rows into in-memory scoring items."""
    items: list[InventoryItem] = []

    for row in records:
        quantity = float(row.get("quantity") or 0)
        expiry = row.get("expiry_date")

        if not expiry:
            continue

        expiry_date = date.fromisoformat(str(expiry)[:10])

        if quantity < 0:
            continue

        inventory_id = str(row["inventory_id"])
        business_id = str(row["business_id"])
        category = str(
            row.get("category") or "uncategorized"
        ).strip().lower()
        product_name = str(
            row.get("product_name") or "Unknown product"
        )
        storage_type = str(
            row.get("storage_type") or "unknown"
        )

        food_type = str(
            row.get("food_type") or ""
        ).strip().lower()

        perishability = (
            "high"
            if food_type == "non-veg"
            or category in {"dairy", "meat", "seafood"}
            else "medium"
        )

        items.append(
            InventoryItem(
                tenant_id=f"foodrescue-business-{business_id}",
                product_id=f"foodrescue-inventory-{inventory_id}",
                batch_id=f"inventory-{inventory_id}",
                name=product_name,
                category=category,
                storage_type=storage_type,
                perishability=perishability,
                current_stock=quantity,
                expiry_date=expiry_date,
                unit_cost=None,
                shelf_life_days=max(
                    1, (expiry_date - date.today()).days
                ),
            )
        )

    return items
