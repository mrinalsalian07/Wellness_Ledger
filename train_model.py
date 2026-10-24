"""
train_model.py
----------------
Generates synthetic personal-finance examples, engineers
financial ratios, and builds a rule-based reference score
(since no public dataset gives us a labeled wellness score directly),
and trains a RandomForestRegressor to learn the mapping from raw
financial features -> wellness score.

Run this once before app.py:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

RNG = np.random.default_rng(42)
N_SAMPLES = 5000

# ---------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------
# We simulate realistic monthly-income individuals (INR) with varying
# financial discipline. "discipline" is a hidden factor (0-1) that
# correlates with good savings/debt behavior, mimicking real population
# heterogeneity without hardcoding the label directly.

def generate_synthetic_data(n=N_SAMPLES):
    discipline = RNG.beta(2, 2, n)  # 0 = poor habits, 1 = great habits

    income = RNG.lognormal(mean=10.5, sigma=0.5, size=n)  # ~monthly INR
    income = np.clip(income, 15000, 300000)

    age = RNG.integers(21, 60, n)
    dependents = RNG.choice([0, 1, 2, 3], size=n, p=[0.35, 0.30, 0.25, 0.10])
    income_sources = RNG.choice([1, 2], size=n, p=[0.78, 0.22])

    # Essential expenses: less disciplined people also spend a higher
    # share on essentials relative to income (less optimization).
    # Include independent household-cost variation and a deliberate tail of
    # people whose expenses exceed income. This broadens training coverage;
    # it does not by itself validate predictions for real people.
    essential_frac = 0.30 + (1 - discipline) * 0.32 + RNG.normal(0, 0.10, n)
    essential_frac = np.clip(essential_frac, 0.15, 1.10)
    discretionary_frac = 0.04 + (1 - discipline) * 0.32 + RNG.normal(0, 0.09, n)
    discretionary_frac = np.clip(discretionary_frac, 0.01, 0.75)
    deficit_mask = RNG.random(n) < 0.12
    essential_frac[deficit_mask] = np.clip(
        essential_frac[deficit_mask] + RNG.uniform(0.15, 0.45, deficit_mask.sum()),
        0.2, 1.2
    )
    essential_expenses = income * essential_frac
    discretionary_expenses = income * discretionary_frac

    # Savings: disciplined people save a much higher share consistently.
    savings_rate_true = 0.02 + discipline * 0.35 + RNG.normal(0, 0.03, n)
    savings_rate_true = np.clip(savings_rate_true, 0.0, 0.6)
    months_saving = RNG.integers(6, 180, n)  # how long they've been saving
    savings_balance = income * savings_rate_true * months_saving * RNG.uniform(0.6, 1.0, n)

    # Debt: less disciplined -> more debt relative to income.
    debt_multiple = 0.2 + (1 - discipline) * 4.2 + RNG.normal(0, 0.8, n)
    high_debt_mask = RNG.random(n) < 0.10
    debt_multiple[high_debt_mask] += RNG.uniform(2.0, 5.0, high_debt_mask.sum())
    debt_multiple = np.clip(debt_multiple, 0, 12)
    total_debt = income * 12 * debt_multiple / 4  # outstanding debt; includes high-debt cases

    debt_repay_frac = 0.05 + (1 - discipline) * 0.35 + RNG.normal(0, 0.03, n)
    debt_repay_frac = np.clip(debt_repay_frac, 0.0, 0.7)
    monthly_debt_repayment = income * debt_repay_frac

    # Emergency fund: correlated with discipline, in months of essential exp.
    emergency_fund_months = discipline * 8 + RNG.normal(0, 0.8, n)
    emergency_fund_months = np.clip(emergency_fund_months, 0, 12)
    emergency_fund = emergency_fund_months * essential_expenses

    df = pd.DataFrame({
        "income": income,
        "essential_expenses": essential_expenses,
        "discretionary_expenses": discretionary_expenses,
        "savings_balance": savings_balance,
        "total_debt": total_debt,
        "monthly_debt_repayment": monthly_debt_repayment,
        "emergency_fund": emergency_fund,
        "income_sources": income_sources,
        "age": age,
        "dependents": dependents,
    })
    return df


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING (shared logic — also imported by app.py)
# ---------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["savings_rate"] = df["savings_balance"] / (df["income"] * 12).clip(lower=1)
    out["savings_rate"] = out["savings_rate"].clip(0, 3)  # allow long-term accumulation

    out["debt_to_income"] = df["total_debt"] / (df["income"] * 12).clip(lower=1)
    out["debt_to_income"] = out["debt_to_income"].clip(0, 10)

    out["debt_repayment_burden"] = df["monthly_debt_repayment"] / df["income"].clip(lower=1)
    out["debt_repayment_burden"] = out["debt_repayment_burden"].clip(0, 1.5)

    total_expenses = df["essential_expenses"] + df["discretionary_expenses"]
    out["expense_to_income"] = total_expenses / df["income"].clip(lower=1)
    out["expense_to_income"] = out["expense_to_income"].clip(0, 2)

    out["emergency_fund_months"] = df["emergency_fund"] / df["essential_expenses"].clip(lower=1)
    out["emergency_fund_months"] = out["emergency_fund_months"].clip(0, 24)

    out["discretionary_ratio"] = df["discretionary_expenses"] / total_expenses.clip(lower=1)
    out["discretionary_ratio"] = out["discretionary_ratio"].clip(0, 1)

    out["income_stability"] = (df["income_sources"] >= 2).astype(int)

    out["dependents"] = df["dependents"]
    out["age"] = df["age"]

    return out


FEATURE_COLUMNS = [
    "savings_rate",
    "debt_to_income",
    "debt_repayment_burden",
    "expense_to_income",
    "emergency_fund_months",
    "discretionary_ratio",
    "income_stability",
]


# ---------------------------------------------------------------------
# 3. RULE-BASED COMPOSITE WELLNESS SCORE (training label)
# ---------------------------------------------------------------------
# This is an illustrative rule-based reference score, not validated ground truth.
# The ML model is trained to approximate these formula-generated labels.
# Evaluation therefore measures agreement with this synthetic scoring system,
# not validated real-world financial outcomes.
def compute_wellness_score(feat: pd.DataFrame, add_noise: bool = False) -> np.ndarray:
    """Deterministic rule score by default; optional noise is used only for training labels."""
    score = np.zeros(len(feat))

    # Savings rate: reward up to 30%, diminishing returns after.
    score += np.clip(feat["savings_rate"] / 0.30, 0, 1) * 25

    # Debt to income (annual): penalize above 1x annual income heavily.
    score += (1 - np.clip(feat["debt_to_income"] / 2.0, 0, 1)) * 20

    # Debt repayment burden: ideal under 30% of monthly income.
    score += (1 - np.clip(feat["debt_repayment_burden"] / 0.5, 0, 1)) * 20

    # Expense to income: ideal under 70%.
    score += (1 - np.clip((feat["expense_to_income"] - 0.3) / 0.9, 0, 1)) * 15

    # Emergency fund: ideal 6+ months.
    score += np.clip(feat["emergency_fund_months"] / 6.0, 0, 1) * 12

    # Discretionary ratio: mild penalty for very high discretionary share.
    score += (1 - np.clip((feat["discretionary_ratio"] - 0.2) / 0.5, 0, 1)) * 3

    # Income stability: small bonus for multiple income sources.
    score += feat["income_stability"] * 5

    if add_noise:
        score = score + RNG.normal(0, 2.5, len(feat))
    score = np.clip(score, 0, 100)
    return score


# ---------------------------------------------------------------------
# 4. TRAIN + EVALUATE + SAVE
# ---------------------------------------------------------------------
def build_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )


def main():
    raw = generate_synthetic_data()
    feat = engineer_features(raw)
    feat["wellness_score"] = compute_wellness_score(feat, add_noise=True)

    X = feat[FEATURE_COLUMNS]
    y = feat["wellness_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae, cv_rmse, cv_r2 = [], [], []

    for train_idx, val_idx in kfold.split(X_train):
        cv_model = build_model()
        cv_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = cv_model.predict(X_train.iloc[val_idx])
        actual = y_train.iloc[val_idx]
        cv_mae.append(mean_absolute_error(actual, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(actual, preds)))
        cv_r2.append(r2_score(actual, preds))

    model = build_model()
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    # Compare learned models and a simple formula baseline on the same hold-out set.
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Mean baseline": DummyRegressor(strategy="mean"),
    }
    baseline_metrics = {}
    for name, baseline in baseline_models.items():
        baseline.fit(X_train, y_train)
        pred = baseline.predict(X_test)
        baseline_metrics[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }
    rule_preds = compute_wellness_score(feat.loc[X_test.index])
    baseline_metrics["Rule-based formula"] = {
        "mae": float(mean_absolute_error(y_test, rule_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, rule_preds))),
        "r2": float(r2_score(y_test, rule_preds)),
    }
    baseline_metrics["Random Forest"] = {
        "mae": float(test_mae), "rmse": float(test_rmse), "r2": float(test_r2)
    }

    print("\n" + "=" * 60)
    print("WELLNESS MODEL EVALUATION")
    print("=" * 60)
    print("\nHold-out test set:")
    print(f"  MAE  : {test_mae:.2f} points")
    print(f"  RMSE : {test_rmse:.2f} points")
    print(f"  R²   : {test_r2:.3f}")
    print("\n5-fold cross-validation on training data:")
    print(f"  MAE  : {np.mean(cv_mae):.2f} +/- {np.std(cv_mae):.2f}")
    print(f"  RMSE : {np.mean(cv_rmse):.2f} +/- {np.std(cv_rmse):.2f}")
    print(f"  R²   : {np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}")
    print("\n" + "=" * 60)

    joblib.dump(model, "wellness_model.joblib")
    joblib.dump(FEATURE_COLUMNS, "feature_columns.joblib")
    with open("model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump({
            "synthetic_samples": int(len(raw)),
            "test_samples": int(len(X_test)),
            "holdout": {"mae": float(test_mae), "rmse": float(test_rmse), "r2": float(test_r2)},
            "baselines": baseline_metrics,
            "cv": {
                "mae_mean": float(np.mean(cv_mae)), "mae_sd": float(np.std(cv_mae)),
                "rmse_mean": float(np.mean(cv_rmse)), "rmse_sd": float(np.std(cv_rmse)),
                "r2_mean": float(np.mean(cv_r2)), "r2_sd": float(np.std(cv_r2)),
            },
            "interpretation": "Targets are synthetic scores generated from a rule-based formula. Metrics describe agreement with that formula-derived target, not validated real-world financial outcomes."
        }, f, indent=2)
    print("Saved wellness_model.joblib, feature_columns.joblib, and model_evaluation.json")


if __name__ == "__main__":
    main()
