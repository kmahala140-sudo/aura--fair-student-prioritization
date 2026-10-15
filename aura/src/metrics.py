"""
Evaluation metrics for ED-01: Fair Student-Support Prioritization.

Implements:
- Capacity-constrained top-20% selection: ceil(0.20 * N)
- Group eligibility filters: >= 10 total rows and >= 3 positive examples
- Group-wise recall: R_g
- Worst-group recall: R_min
- Attribute fairness gaps and overall fairness gap
- Brier score and calibration utility
- Official 100-point judging criteria score
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_top_k_selection(
    probabilities: np.ndarray,
    budget_fraction: float = 0.20,
    tie_breaker: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Select exactly ceil(budget_fraction * N) students by ranking descending by probability.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Array of predicted probabilities of support needed.
    budget_fraction : float, default=0.20
        Fraction of students to select.
    tie_breaker : np.ndarray, optional
        Optional secondary key for stable deterministic tie-breaking.
        
    Returns
    -------
    selected_mask : np.ndarray (bool)
        Boolean array of length N where True indicates selection for support.
    """
    n = len(probabilities)
    if n == 0:
        return np.array([], dtype=bool)
    
    k = math.ceil(budget_fraction * n)
    
    # Use stable sort with tie_breaker if provided
    if tie_breaker is not None:
        order = np.lexsort((-tie_breaker, -probabilities))
    else:
        # np.argsort on negative probabilities (stable sort)
        order = np.argsort(-probabilities, kind="mergesort")
        
    selected_indices = set(order[:k])
    selected_mask = np.zeros(n, dtype=bool)
    for idx in selected_indices:
        selected_mask[idx] = True
        
    return selected_mask


def evaluate_fair_prioritization(
    df: pd.DataFrame,
    prob_col: str = "support_probability",
    target_col: str = "support_needed",
    protected_attributes: Optional[List[str]] = None,
    min_group_size: int = 10,
    min_group_positives: int = 3,
    budget_fraction: float = 0.20,
    reproducibility_points: float = 5.0
) -> Dict[str, Any]:
    """
    Calculates the full suite of utility, fairness, and calibration metrics
    matching the ED-01 competition evaluation specification.
    """
    if protected_attributes is None:
        protected_attributes = ["sex", "school"]

    n_total = len(df)
    if n_total == 0:
        raise ValueError("Cannot evaluate an empty dataframe.")

    k_selected = math.ceil(budget_fraction * n_total)
    
    # 1. Rank and select exactly ceil(0.20 * N)
    df_eval = df.copy().reset_index(drop=True)
    probs = df_eval[prob_col].to_numpy(dtype=float)
    
    # Deterministic selection
    selected_mask = compute_top_k_selection(probs, budget_fraction=budget_fraction)
    df_eval["_selected"] = selected_mask
    df_eval["_rank"] = np.argsort(np.argsort(-probs, kind="mergesort")) + 1

    # 2. Calibration & Brier Score
    y_true = df_eval[target_col].to_numpy(dtype=int)
    brier_score = float(np.mean((probs - y_true) ** 2))
    calib_utility = float(1.0 - brier_score)

    # 3. Overall Recall
    total_positives = int(np.sum(y_true == 1))
    selected_positives = int(np.sum((df_eval["_selected"]) & (y_true == 1)))
    overall_recall = float(selected_positives / total_positives) if total_positives > 0 else 0.0
    overall_precision = float(selected_positives / k_selected) if k_selected > 0 else 0.0

    # 4. Group-wise Evaluation
    group_stats: List[Dict[str, Any]] = []
    eligible_group_recalls: Dict[str, float] = {}
    attribute_gaps: Dict[str, Optional[float]] = {}

    for attr in protected_attributes:
        if attr not in df_eval.columns:
            continue
            
        unique_groups = sorted(df_eval[attr].unique())
        attr_eligible_recalls: Dict[str, float] = {}

        for g in unique_groups:
            sub = df_eval[df_eval[attr] == g]
            n_rows = len(sub)
            n_pos = int(np.sum(sub[target_col] == 1))
            n_sel = int(np.sum(sub["_selected"]))
            n_sel_pos = int(np.sum((sub["_selected"]) & (sub[target_col] == 1)))

            # Eligibility rule: >= 10 rows and >= 3 positive examples
            is_eligible = (n_rows >= min_group_size) and (n_pos >= min_group_positives)
            recall = float(n_sel_pos / n_pos) if n_pos > 0 else 0.0
            precision = float(n_sel_pos / n_sel) if n_sel > 0 else 0.0
            selection_rate = float(n_sel / n_rows) if n_rows > 0 else 0.0

            group_key = f"{attr}:{g}"
            if is_eligible:
                eligible_group_recalls[group_key] = recall
                attr_eligible_recalls[str(g)] = recall

            group_stats.append({
                "attribute": attr,
                "group": str(g),
                "total_rows": n_rows,
                "positive_rows": n_pos,
                "selected_rows": n_sel,
                "selected_positive_rows": n_sel_pos,
                "is_eligible": is_eligible,
                "recall": recall,
                "precision": precision,
                "selection_rate": selection_rate
            })

        # Calculate gap within attribute if at least 2 eligible groups exist
        if len(attr_eligible_recalls) >= 2:
            vals = list(attr_eligible_recalls.values())
            # Largest absolute recall difference among eligible groups of this attribute
            max_diff = max(vals) - min(vals)
            attribute_gaps[attr] = float(max_diff)
        else:
            attribute_gaps[attr] = None

    # 5. Worst-Group Recall (R_min)
    if len(eligible_group_recalls) > 0:
        r_min = float(min(eligible_group_recalls.values()))
        r_min_group = min(eligible_group_recalls, key=eligible_group_recalls.get)
    else:
        r_min = overall_recall
        r_min_group = "N/A"

    # 6. Overall Fairness Gap
    # "If fewer than two eligible groups exist for an attribute, that attribute contributes no gap.
    # If neither attribute has two eligible groups, the fairness gap is 0."
    valid_gaps = [gap for gap in attribute_gaps.values() if gap is not None]
    fairness_gap = float(max(valid_gaps)) if valid_gaps else 0.0
    fairness_utility = float(1.0 - fairness_gap)

    # 7. Total Competition Score (out of 100)
    # 40 pts - recall within top-20%
    # 25 pts - worst eligible-group recall Rmin
    # 20 pts - fairness utility (1 - fairness gap)
    # 10 pts - calibration utility (1 - Brier)
    # 5 pts - reproducibility
    points_recall = 40.0 * overall_recall
    points_rmin = 25.0 * r_min
    points_fairness = 20.0 * fairness_utility
    points_calibration = 10.0 * calib_utility
    points_reproducibility = reproducibility_points

    total_score = (
        points_recall +
        points_rmin +
        points_fairness +
        points_calibration +
        points_reproducibility
    )

    return {
        "total_score": round(total_score, 4),
        "overall_recall": round(overall_recall, 4),
        "worst_group_recall": round(r_min, 4),
        "worst_group_name": r_min_group,
        "fairness_gap": round(fairness_gap, 4),
        "fairness_utility": round(fairness_utility, 4),
        "brier_score": round(brier_score, 4),
        "calibration_utility": round(calib_utility, 4),
        "points_breakdown": {
            "recall_points": round(points_recall, 4),
            "worst_group_recall_points": round(points_rmin, 4),
            "fairness_utility_points": round(points_fairness, 4),
            "calibration_utility_points": round(points_calibration, 4),
            "reproducibility_points": round(points_reproducibility, 4),
        },
        "k_selected": k_selected,
        "n_total": n_total,
        "total_positives": total_positives,
        "selected_positives": selected_positives,
        "overall_precision": round(overall_precision, 4),
        "attribute_gaps": attribute_gaps,
        "eligible_group_recalls": {k: round(v, 4) for k, v in eligible_group_recalls.items()},
        "group_stats": pd.DataFrame(group_stats)
    }


def format_fairness_table(results: Dict[str, Any]) -> str:
    """Returns a clean markdown table formatting group statistics."""
    df_groups = results["group_stats"]
    lines = [
        "| Attribute | Group | Total Rows | Positive Rows | Selected Rows | Eligible? | Group Recall | Precision | Selection Rate |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for _, row in df_groups.iterrows():
        elig_str = "Yes" if row["is_eligible"] else "No"
        rec_str = f"{row['recall'] * 100:.1f}%"
        prec_str = f"{row['precision'] * 100:.1f}%"
        sel_rate_str = f"{row['selection_rate'] * 100:.1f}%"
        lines.append(
            f"| {row['attribute']} | {row['group']} | {row['total_rows']} | {row['positive_rows']} | "
            f"{row['selected_rows']} | {elig_str} | {rec_str} | {prec_str} | {sel_rate_str} |"
        )
    return "\n".join(lines)
