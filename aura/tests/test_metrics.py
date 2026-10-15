"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest
from src.metrics import compute_top_k_selection, evaluate_fair_prioritization


def test_top_k_selection_exact_ceil():
    # 10 items -> ceil(0.20 * 10) = 2
    probs = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.3, 0.5, 0.7, 0.6, 0.05])
    sel = compute_top_k_selection(probs, budget_fraction=0.20)
    assert np.sum(sel) == 2
    # Indices of 0.9 and 0.8 should be True
    assert sel[1] == True
    assert sel[3] == True

    # 11 items -> ceil(0.20 * 11) = 3
    probs_11 = np.append(probs, 0.15)
    sel_11 = compute_top_k_selection(probs_11, budget_fraction=0.20)
    assert np.sum(sel_11) == 3


def test_eligibility_and_fairness_gap():
    # Create synthetic dataset with:
    # Group GP: 15 rows, 5 positives (eligible)
    # Group MS: 8 rows, 2 positives (ineligible due to size < 10 and pos < 3)
    # Group F: 12 rows, 4 positives (eligible)
    # Group M: 11 rows, 3 positives (eligible)
    
    rows = []
    # GP: 15 rows (5 positives)
    for i in range(15):
        rows.append({
            "school": "GP",
            "sex": "F" if i % 2 == 0 else "M",
            "support_needed": 1 if i < 5 else 0,
            "support_probability": 0.8 if i < 5 else 0.1
        })
    # MS: 8 rows (2 positives)
    for i in range(8):
        rows.append({
            "school": "MS",
            "sex": "F" if i % 2 == 0 else "M",
            "support_needed": 1 if i < 2 else 0,
            "support_probability": 0.7 if i < 2 else 0.1
        })
    df = pd.DataFrame(rows)
    
    res = evaluate_fair_prioritization(df, prob_col="support_probability", target_col="support_needed")
    
    # Check MS is ineligible for group metrics
    group_stats = res["group_stats"]
    ms_stat = group_stats[group_stats["group"] == "MS"].iloc[0]
    assert ms_stat["is_eligible"] == False

    gp_stat = group_stats[group_stats["group"] == "GP"].iloc[0]
    assert gp_stat["is_eligible"] == True

    # Because only GP is eligible for school, school gap should be None, and overall gap is sex gap
    assert res["attribute_gaps"]["school"] is None
    assert "sex" in res["attribute_gaps"]
    assert res["fairness_gap"] >= 0.0


def test_perfect_prediction_score():
    # 20 rows, 4 positives (all in top 4 -> 100% recall of budget ceil(0.20*20) = 4)
    # Balanced groups: GP (10 rows, 2 pos), MS (10 rows, 2 pos)
    # sex: F (10 rows, 2 pos), M (10 rows, 2 pos)
    rows = []
    for i in range(20):
        is_pos = (i < 4)
        rows.append({
            "school": "GP" if i % 2 == 0 else "MS",
            "sex": "F" if (i // 2) % 2 == 0 else "M",
            "support_needed": 1 if is_pos else 0,
            "support_probability": 0.95 if is_pos else 0.05
        })
    df = pd.DataFrame(rows)
    
    res = evaluate_fair_prioritization(df, prob_col="support_probability", target_col="support_needed")
    # All 4 positives selected
    assert res["overall_recall"] == 1.0
    assert res["k_selected"] == 4
    # Brier score should be very low
    assert res["brier_score"] < 0.01
    # Check point components
    pb = res["points_breakdown"]
    assert pb["recall_points"] == 40.0
    assert pb["reproducibility_points"] == 5.0
