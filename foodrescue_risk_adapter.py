
"""Read-only bridge from FoodRescue inventory to waste-risk scoring."""

from datetime import date
from typing import Any

from .db import InventoryItem


def normalize_category(value: Any) -> str:
    """Standardize FoodRescue category names."""
    category = str(value or "uncategorized").strip().lower()

    aliases = {
        "vegetable": "vegetables",
        "fruit": "fruits",
        "grain": "grains",
    }

    return aliases.get(category, category)


def normalize_storage(value: Any) -> str:
    """Map FoodRescue storage labels to AI engine storage types."""
    storage = str(value or "").strip().lower()

    if storage in {
        "refrigerated",
        "refrigerator",
        "refrierator",
        "chilled",
    }:
        return "chilled"

    if storage in {"frozen", "freezer"}:
        return "frozen"

    return "ambient"


def foodrescue_inventory_to_items(
    records: list[dict[str, Any]],
) -> list[InventoryItem]:
    """Convert FoodRescue inventory rows into in-memory scoring items."""
    items: list[InventoryItem] = []

    for row in records:
        try:
            quantity = float(row.get("quantity") or 0)
            expiry = row.get("expiry_date")

            if not expiry or quantity < 0:
                continue

            expiry_date = date.fromisoformat(str(expiry)[:10])

            inventory_id = str(row["inventory_id"])
            business_id = str(row["business_id"])

            category = normalize_category(row.get("category"))

            product_name = str(
                row.get("product_name") or "Unknown product"
            ).strip()

            storage_type = normalize_storage(
                row.get("storage_type")
            )

            food_type = str(
                row.get("food_type") or ""
            ).strip().lower()

            perishability = (
                "high"
                if food_type in {"non-veg", "non veg", "nonveg"}
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
                        1,
                        (
                            expiry_date - date.today()
                        ).days
                    ),
                )
            )

        except (KeyError, TypeError, ValueError):
            continue

    return items
