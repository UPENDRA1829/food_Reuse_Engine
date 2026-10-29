
"""Read-only adapter for the FoodRescue SQLite database.

This module reads inventory and sales for integration checks.
It does not modify the FoodRescue database or seed synthetic data.
"""

import os
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_FOODRESCUE_DB = (
    r"D:\Food_Redistribute_Project\backend\food_waste.db"
)


def normalize_category(value: object) -> str:
    """Normalize category labels without changing source data."""
    if value is None:
        return ""

    normalized = " ".join(str(value).strip().casefold().split())

    aliases = {
        "fruit": "fruits",
        "fruits": "fruits",
        "vegetable": "vegetables",
        "vegetables": "vegetables",
        "grain": "grains",
        "grains": "grains",
        "dairy": "dairy",
        "meat": "meat",
        "bakery": "bakery",
    }

    return aliases.get(normalized, normalized)


def normalize_unit(value: object) -> str:
    """Normalize common unit labels without converting quantities."""
    if value is None:
        return ""

    normalized = " ".join(str(value).strip().casefold().split())

    aliases = {
        "kg": "kg",
        "kgs": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "liter": "liters",
        "liters": "liters",
        "litre": "liters",
        "litres": "liters",
        "iters": "liters",
        "pack": "pack",
        "piece": "piece",
    }

    return aliases.get(normalized, normalized)


def _database_path() -> Path:
    """Return the database path, allowing an environment override."""
    configured = os.getenv(
        "FOODRESCUE_DB_PATH",
        DEFAULT_FOODRESCUE_DB,
    )
    path = Path(configured)

    if not path.is_file():
        raise FileNotFoundError(
            f"FoodRescue database not found: {path}"
        )

    return path


def _connect_read_only() -> sqlite3.Connection:
    """Open the FoodRescue SQLite database in read-only mode."""
    path = _database_path()
    uri = path.resolve().as_uri() + "?mode=ro"

    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def get_inventory(limit: int = 100) -> list[dict[str, Any]]:
    """Read current inventory without changing the database."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    with _connect_read_only() as connection:
        rows = connection.execute(
            """
            SELECT
                inventory_id,
                business_id,
                product_name,
                category,
                quantity,
                unit,
                purchase_date,
                expiry_date,
                storage_type,
                food_type
            FROM inventory
            ORDER BY inventory_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_sales(limit: int = 100) -> list[dict[str, Any]]:
    """Read sales records without changing the database."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    with _connect_read_only() as connection:
        rows = connection.execute(
            """
            SELECT
                sale_id,
                inventory_id,
                quantity_sold,
                unit,
                total_amount,
                payment_method,
                sold_at
            FROM sales
            ORDER BY sale_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_summary() -> dict[str, Any]:
    """Return record counts and data-readiness warnings."""
    with _connect_read_only() as connection:
        inventory_count = connection.execute(
            "SELECT COUNT(*) FROM inventory"
        ).fetchone()[0]

        sales_count = connection.execute(
            "SELECT COUNT(*) FROM sales"
        ).fetchone()[0]

        business_count = connection.execute(
            "SELECT COUNT(*) FROM businesses"
        ).fetchone()[0]

    return {
        "source": "FoodRescue",
        "database_mode": "read-only",
        "business_count": business_count,
        "inventory_count": inventory_count,
        "sales_count": sales_count,
        "purchase_history": (
            "Inventory purchase dates are available; "
            "a complete purchase transaction history is not confirmed."
        ),
        "waste_history": (
            "Actual waste records have not yet been identified."
        ),
        "ml_readiness": (
            "Integration check only; sales history must be assessed "
            "before training or trusting demand forecasts."
        ),
    }


def get_sales_history(limit: int = 1000) -> list[dict[str, Any]]:
    """Read sales joined to inventory details."""
    if limit < 1 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000")

    with _connect_read_only() as connection:
        rows = connection.execute(
            """
            SELECT
                s.sale_id,
                s.inventory_id,
                i.product_name,
                i.category,
                s.quantity_sold,
                s.unit,
                s.sold_at
            FROM sales AS s
            INNER JOIN inventory AS i
                ON i.inventory_id = s.inventory_id
            ORDER BY s.sold_at, s.sale_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_daily_sales_history(
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Aggregate real sales by category, date, and normalized unit."""
    sales = get_sales_history(limit=limit)

    # Include unit in the key to avoid combining incompatible quantities.
    daily_totals: dict[tuple[str, str, str], float] = defaultdict(float)

    for sale in sales:
        sold_at = sale.get("sold_at")
        category = sale.get("category")
        quantity = sale.get("quantity_sold")

        if not sold_at or not category or quantity is None:
            continue

        sale_date = str(sold_at)[:10]

        try:
            date.fromisoformat(sale_date)
            quantity_value = float(quantity)
        except (ValueError, TypeError):
            continue

        if quantity_value < 0:
            continue

        normalized_category = normalize_category(category)
        normalized_unit = normalize_unit(sale.get("unit"))

        if not normalized_category:
            continue

        daily_totals[
            (normalized_category, sale_date, normalized_unit)
        ] += quantity_value

    return [
        {
            "category": category,
            "sale_date": sale_date,
            "quantity": quantity,
            "unit": unit,
        }
        for (category, sale_date, unit), quantity
        in sorted(daily_totals.items())
    ]
