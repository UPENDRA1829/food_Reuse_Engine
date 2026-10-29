
"""Import real FoodRescue sales into the AI database.

FoodRescue source database remains read-only.
Sales are grouped by business and inventory product.
"""

import json
from datetime import datetime

from sqlalchemy import delete

from .db import SaleRecord, session_scope
from .foodrescue_adapter import get_inventory, get_sales_history


def normalize_category(value):
    category = str(value or "uncategorized").strip().lower()
    if category == "vegetable":
        return "vegetables"
    return category


def import_real_sales():
    sales = get_sales_history(1000)
    inventory = get_inventory(1000)

    # Match each inventory record to its business.
    inventory_business = {
        int(row["inventory_id"]): int(row["business_id"])
        for row in inventory
    }

    valid = []
    skipped = 0
    tenants = set()

    for row in sales:
        try:
            inventory_id = int(row["inventory_id"])
            business_id = inventory_business.get(inventory_id)

            if business_id is None:
                skipped += 1
                continue

            quantity = float(row["quantity_sold"])
            category = normalize_category(row["category"])
            sale_date = datetime.fromisoformat(
                str(row["sold_at"])
            ).date()

            if quantity <= 0:
                skipped += 1
                continue

            tenant_id = f"foodrescue-business-{business_id}"
            tenants.add(tenant_id)

            valid.append({
                "tenant_id": tenant_id,
                "product_id": f"foodrescue-inventory-{inventory_id}",
                "category": category,
                "sale_date": sale_date,
                "quantity": quantity,
            })

        except (KeyError, TypeError, ValueError):
            skipped += 1

    # Update only FoodRescue business tenants.
    # Preserve demonstration data and unrelated tenants.
    with session_scope() as session:
        for tenant_id in tenants:
            session.execute(
                delete(SaleRecord).where(
                    SaleRecord.tenant_id == tenant_id
                )
            )

        for record in valid:
            session.add(SaleRecord(**record))

    return {
        "source": "FoodRescue",
        "imported_transactions": len(valid),
        "skipped_transactions": skipped,
        "business_tenants": sorted(tenants),
        "training_ready": False,
        "message": (
            "Real sales imported with matching inventory IDs. "
            "Additional historical sales are required for "
            "reliable Prophet/LSTM forecasting."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(import_real_sales(), indent=2))
