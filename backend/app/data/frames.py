"""Load stored tables and build the leakage-safe analysis frame (one row per customer decision)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import settings
from .schema import ARMS, FEATURES, NEGATIVE_CONTROL_OUTCOME, OUTCOMES, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID

TABLES = ["customers", "loans", "repayments", "interventions", "outcomes", "economic_indicators", "ground_truth"]


def dataset_dir(name: str) -> Path:
    base = settings.synthetic_dir
    p = (base / name).resolve()
    if base.resolve() not in p.parents:  # restrict artifact paths
        raise ValueError("invalid dataset name")
    return p


def load_tables(name: str) -> dict[str, pd.DataFrame]:
    d = dataset_dir(name)
    if not d.exists():
        raise FileNotFoundError(name)
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES if (d / f"{t}.parquet").exists()}


def manifest(name: str) -> dict:
    return json.loads((dataset_dir(name) / "manifest.json").read_text(encoding="utf-8"))


def build_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge pre-treatment covariates with assignment and outcomes.

    Only information dated on/before the decision date enters feature columns; the outcome window
    starts the day after (validated in `validate_temporal`).
    """
    c, l, i, o, e = (tables[k] for k in ("customers", "loans", "interventions", "outcomes", "economic_indicators"))
    econ = e[["cohort_month", "regional_economic_group", "economic_stress_index"]].rename(
        columns={"regional_economic_group": "region_group"})
    df = c.merge(l, on="customer_id", how="left")
    df = df.merge(i[["customer_id", "intervention_type", "intervention_cost", "delivery_status", "assignment_probability",
                     "agent_or_channel_group"]].rename(columns={"intervention_type": "arm"}), on="customer_id", how="left")
    df = df.merge(o[["customer_id", "outcome_window_start", "outcome_window_end", "observation_complete"] + OUTCOMES + ["repayment_amount"]],
                  on="customer_id", how="left")
    df = df.merge(econ, on=["cohort_month", "region_group"], how="left")
    df["log_monthly_income"] = np.log(df["monthly_income"])
    df["expense_ratio"] = df["estimated_monthly_expenses"] / df["monthly_income"]
    df["installment_to_income"] = df["installment_amount"] / df["monthly_income"]
    df["decision_date"] = pd.to_datetime(df["data_snapshot_date"])
    df["split"] = np.where(df["cohort_month"].isin(list(SPLIT_TRAIN)), "train",
                           np.where(df["cohort_month"].isin(list(SPLIT_VALID)), "validation", "test"))
    # eligibility flags mirror stored rules (derived from observed fields at time zero)
    df["elig_standard_reminder"] = df["eligibility_status"].eq("contactable")
    df["elig_counselling"] = df["elig_standard_reminder"] & ((df["historical_delinquency_count"] >= 1)
                              | (df["financial_stress_indicator"] > 0.5) | (df["debt_to_income_ratio"] > 0.40))
    df["elig_repayment_plan"] = (df["loan_type"] != "credit_line") & (df["remaining_term_months"] >= 6) & (
        (df["debt_to_income_ratio"] >= 0.35) | (df["historical_delinquency_count"] >= 2))
    df["elig_none"] = True
    assert set(df["arm"].unique()) <= set(ARMS)
    return df.reset_index(drop=True)


@lru_cache(maxsize=4)
def get_frame(name: str) -> pd.DataFrame:
    return build_frame(load_tables(name))


@lru_cache(maxsize=4)
def get_truth(name: str) -> pd.DataFrame:
    return load_tables(name)["ground_truth"]


def clear_cache() -> None:
    get_frame.cache_clear()
    get_truth.cache_clear()


def leakage_free_features() -> list[str]:
    forbidden = set(OUTCOMES) | {NEGATIVE_CONTROL_OUTCOME, "arm", "intervention_cost", "delivery_status", "repayment_amount",
                                 "assignment_probability"}
    assert not (set(FEATURES) & forbidden)
    return FEATURES
