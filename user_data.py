
"""
user_data.py
------------
Persistent storage for Wellness Ledger monthly check-ins and goals.

Uses a small local SQLite database. It is keyed by the authenticated
username, so each user's check-ins and goals stay separate.
"""

import sqlite3
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DB_PATH = Path(__file__).resolve().parent / "wellness_ledger_data.db"
IST = ZoneInfo("Asia/Kolkata")


def _now():
    return datetime.now(IST)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _connection():
    """Commit on success, roll back on failure, and always close the connection."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                month TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                income REAL NOT NULL,
                essential_expenses REAL NOT NULL,
                discretionary_expenses REAL NOT NULL,
                savings_balance REAL NOT NULL,
                total_debt REAL NOT NULL,
                monthly_debt_repayment REAL NOT NULL,
                emergency_fund REAL NOT NULL,
                income_sources INTEGER NOT NULL,
                age INTEGER NOT NULL,
                dependents INTEGER NOT NULL,
                score REAL NOT NULL,
                top_factor TEXT,
                UNIQUE(username, month)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL NOT NULL DEFAULT 0,
                target_date TEXT,
                created_at TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def current_month():
    return _now().strftime("%Y-%m")


def current_month_label():
    return _now().strftime("%B %Y")


def save_checkin(username, month, data, score, top_factor):
    """Validate and save one monthly snapshot for an authenticated user."""
    username = str(username).strip().lower()
    if not username:
        raise ValueError("A signed-in username is required.")
    if not isinstance(month, str) or len(month) != 7:
        raise ValueError("Month must use YYYY-MM format.")
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("Month must use YYYY-MM format.") from exc

    numeric_fields = (
        "income", "essential_expenses", "discretionary_expenses",
        "savings_balance", "total_debt", "monthly_debt_repayment",
        "emergency_fund",
    )
    values = {}
    for field in numeric_fields:
        try:
            value = float(data[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field.replace('_', ' ').title()} must be a number.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("Financial amounts cannot be negative or non-finite.")
        values[field] = value
    if values["income"] <= 0:
        raise ValueError("Monthly income must be greater than ₹0.")
    if values["monthly_debt_repayment"] > values["income"] * 5:
        raise ValueError("Monthly debt repayments look unusually high. Please check the amount.")
    for field, low, high in (("income_sources", 1, 3), ("age", 18, 100), ("dependents", 0, 20)):
        try:
            value = int(data[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field.replace('_', ' ').title()} is invalid.") from exc
        if value < low or value > high:
            raise ValueError(f"{field.replace('_', ' ').title()} must be between {low} and {high}.")
    if not math.isfinite(float(score)) or not 0 <= float(score) <= 100:
        raise ValueError("The wellness score must be between 0 and 100.")

    now = _now().isoformat(timespec="seconds")

    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO checkins (
                username, month, created_at, updated_at,
                income, essential_expenses, discretionary_expenses,
                savings_balance, total_debt, monthly_debt_repayment,
                emergency_fund, income_sources, age, dependents,
                score, top_factor
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, month) DO UPDATE SET
                updated_at = excluded.updated_at,
                income = excluded.income,
                essential_expenses = excluded.essential_expenses,
                discretionary_expenses = excluded.discretionary_expenses,
                savings_balance = excluded.savings_balance,
                total_debt = excluded.total_debt,
                monthly_debt_repayment = excluded.monthly_debt_repayment,
                emergency_fund = excluded.emergency_fund,
                income_sources = excluded.income_sources,
                age = excluded.age,
                dependents = excluded.dependents,
                score = excluded.score,
                top_factor = excluded.top_factor
            """,
            (
                username,
                month,
                now,
                now,
                data["income"],
                data["essential_expenses"],
                data["discretionary_expenses"],
                data["savings_balance"],
                data["total_debt"],
                data["monthly_debt_repayment"],
                data["emergency_fund"],
                data["income_sources"],
                data["age"],
                data["dependents"],
                score,
                top_factor,
            ),
        )


def get_checkin(username, month=None):
    month = month or current_month()

    with _connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM checkins
            WHERE username = ? AND month = ?
            """,
            (username, month),
        ).fetchone()

    return dict(row) if row else None


def get_checkins(username, limit=12):
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM checkins
            WHERE username = ?
            ORDER BY month DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def create_goal(
    username,
    name,
    category,
    target_amount,
    current_amount=0,
    target_date=None,
):
    username = str(username).strip().lower()
    name = str(name).strip()
    category = str(category).strip()
    if not username:
        raise ValueError("A signed-in username is required.")
    if not name or len(name) > 50:
        raise ValueError("Goal name must contain 1–50 characters.")
    if not category or len(category) > 40:
        raise ValueError("Please choose a valid goal category.")
    current_amount = float(current_amount)
    target_amount = float(target_amount)
    if not math.isfinite(current_amount) or current_amount < 0:
        raise ValueError("Already-saved amount cannot be negative.")
    if not math.isfinite(target_amount) or target_amount <= 0:
        raise ValueError("Target amount must be greater than ₹0.")
    if current_amount > 1_000_000_000_000 or target_amount > 1_000_000_000_000:
        raise ValueError("Please enter an amount within the supported range.")
    now = _now().isoformat(timespec="seconds")
    completed = 1 if current_amount >= target_amount else 0

    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO goals (
                username, name, category, target_amount,
                current_amount, target_date, created_at, completed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                name.strip(),
                category,
                target_amount,
                current_amount,
                target_date,
                now,
                completed,
            ),
        )
        return cursor.lastrowid


def get_goals(username, include_completed=True):
    query = """
        SELECT * FROM goals
        WHERE username = ?
    """
    params = [username]

    if not include_completed:
        query += " AND completed = 0"

    query += " ORDER BY completed ASC, id DESC"

    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def update_goal_progress(username, goal_id, current_amount):
    current_amount = max(0.0, float(current_amount))

    with _connection() as conn:
        row = conn.execute(
            """
            SELECT target_amount FROM goals
            WHERE id = ? AND username = ?
            """,
            (goal_id, username),
        ).fetchone()

        if not row:
            return False

        completed = 1 if current_amount >= row["target_amount"] else 0

        conn.execute(
            """
            UPDATE goals
            SET current_amount = ?, completed = ?
            WHERE id = ? AND username = ?
            """,
            (current_amount, completed, goal_id, username),
        )

    return True


def delete_goal(username, goal_id):
    with _connection() as conn:
        conn.execute(
            """
            DELETE FROM goals
            WHERE id = ? AND username = ?
            """,
            (goal_id, username),
        )



def delete_user_data(username):
    """Delete a user's check-ins and goals. Account credentials are managed separately."""
    username = str(username).strip().lower()
    if not username:
        raise ValueError("A signed-in username is required.")
    with _connection() as conn:
        conn.execute("DELETE FROM checkins WHERE username = ?", (username,))
        conn.execute("DELETE FROM goals WHERE username = ?", (username,))
