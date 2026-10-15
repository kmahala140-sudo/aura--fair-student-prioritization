"""
Training and cross-validation pipeline for ED-01: Fair Student-Support Prioritization.

Usage:
    python train.py --data data/student-mat.csv --model_out models/fair_student_model.pkl --output_dir outputs/
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from src.metrics import evaluate_fair_prioritization, format_fairness_table
from src.model import FairStudentPrioritizer, extract_features_and_target


def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads CSV with automatic delimiter detection (; or ,)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Try semicolon first (standard for student-mat.csv)
    try:
        df = pd.read_csv(filepath, sep=";")
        if df.shape[1] > 5:
            return df
    except Exception:
        pass

    # Fallback to comma
    return pd.read_csv(filepath, sep=",")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Fair Student Prioritizer.")
    parser.add_argument("--data", type=str, default="data/student-mat.csv", help="Path to student-mat.csv")
    parser.add_argument("--model_out", type=str, default="models/fair_student_model.pkl", help="Model artifact path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for predictions and tables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--budget", type=float, default=0.20, help="Support selection capacity fraction (default 0.20)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ED-01: Fair Student-Support Prioritization — Training & Validation")
    print("=" * 70)
    print(f"Loading development dataset from: {args.data}")
    df_raw = load_dataset(args.data)
    n_rows, n_cols = df_raw.shape
    print(f"Dataset loaded: {n_rows} rows, {n_cols} columns")

    # Define target support_needed = 1 if G3 < 10 else 0
    if "G3" in df_raw.columns and "support_needed" not in df_raw.columns:
        df_raw["support_needed"] = (df_raw["G3"] < 10).astype(int)

    # Separate X, y, ids. Guarantee G3 is dropped from features.
    X, y, ids = extract_features_and_target(df_raw)
    assert "G3" not in X.columns, "CRITICAL ERROR: G3 leaked into feature matrix!"
    assert "support_needed" not in X.columns, "CRITICAL ERROR: target leaked into feature matrix!"
    print(f"Permitted feature matrix X shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (support_needed=1: {np.sum(y)} / {len(y)} [{np.mean(y)*100:.1f}%])")

    # 1. Evaluate Unconstrained Baseline Model (no fair alignment)
    print("\n" + "-" * 70)
    print("Phase 1: Evaluating Baseline Model (Raw Unconstrained Ensemble)")
    print("-" * 70)
    baseline_model = FairStudentPrioritizer(
        enable_fair_alignment=False,
        random_state=args.seed
    )
    baseline_model.fit(X, y)
    df_base_eval = df_raw.copy()
    # Re-evaluate baseline with exact metrics
    df_base_eval["support_probability"] = baseline_model.predict_proba(X)
    metrics_baseline = evaluate_fair_prioritization(
        df_base_eval,
        prob_col="support_probability",
        target_col="support_needed",
        budget_fraction=args.budget
    )

    print(f"Baseline Overall Recall @ 20%: {metrics_baseline['overall_recall'] * 100:.2f}% "
          f"({metrics_baseline['selected_positives']}/{metrics_baseline['k_selected']} selected)")
    print(f"Baseline Worst-Group Recall:    {metrics_baseline['worst_group_recall'] * 100:.2f}% "
          f"(Group: {metrics_baseline['worst_group_name']})")
    print(f"Baseline Fairness Gap:         {metrics_baseline['fairness_gap'] * 100:.2f}% "
          f"(Sex: {metrics_baseline['attribute_gaps'].get('sex', 0)*100:.2f}%, "
          f"School: {metrics_baseline['attribute_gaps'].get('school', 0)*100:.2f}%)")
    print(f"Baseline Brier Score:          {metrics_baseline['brier_score']:.4f} "
          f"(Utility: {metrics_baseline['calibration_utility']:.4f})")
    print(f"Baseline Total Score:          {metrics_baseline['total_score']:.2f} / 100")

    # 2. Train and Evaluate Fair-Aligned Prioritization Model
    print("\n" + "-" * 70)
    print("Phase 2: Training & Validating Fair Prioritizer (Pareto Alignment)")
    print("-" * 70)
    fair_model = FairStudentPrioritizer(
        enable_fair_alignment=True,
        random_state=args.seed
    )
    fair_model.fit(X, y)
    print(f"Learned in-fold Pareto alignment parameters:")
    print(f"  * School (GP offset alpha): {fair_model.alpha_school_:+.4f}")
    print(f"  * Sex (F offset beta):      {fair_model.beta_sex_:+.4f}")

    df_fair_eval = df_raw.copy()
    df_fair_eval["support_probability"] = fair_model.predict_proba(X)
    metrics_fair = evaluate_fair_prioritization(
        df_fair_eval,
        prob_col="support_probability",
        target_col="support_needed",
        budget_fraction=args.budget
    )

    print(f"Fair Model Overall Recall @ 20%: {metrics_fair['overall_recall'] * 100:.2f}% "
          f"({metrics_fair['selected_positives']}/{metrics_fair['k_selected']} selected)")
    print(f"Fair Model Worst-Group Recall:    {metrics_fair['worst_group_recall'] * 100:.2f}% "
          f"(Group: {metrics_fair['worst_group_name']})")
    print(f"Fair Model Fairness Gap:         {metrics_fair['fairness_gap'] * 100:.2f}% "
          f"(Sex: {metrics_fair['attribute_gaps'].get('sex', 0)*100:.2f}%, "
          f"School: {metrics_fair['attribute_gaps'].get('school', 0)*100:.2f}%)")
    print(f"Fair Model Brier Score:          {metrics_fair['brier_score']:.4f} "
          f"(Utility: {metrics_fair['calibration_utility']:.4f})")
    print(f"Fair Model Total Score:          {metrics_fair['total_score']:.2f} / 100")

    # 3. Model Comparison Table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Judging Criteria: 100 Points)")
    print("=" * 70)
    comp_df = pd.DataFrame([
        {
            "Model": "Baseline (Unconstrained)",
            "Recall @ 20% (40pt)": f"{metrics_baseline['overall_recall']*100:.2f}% ({metrics_baseline['points_breakdown']['recall_points']:.2f})",
            "Worst Group Rmin (25pt)": f"{metrics_baseline['worst_group_recall']*100:.2f}% ({metrics_baseline['points_breakdown']['worst_group_recall_points']:.2f})",
            "Fairness Gap (20pt)": f"{metrics_baseline['fairness_gap']*100:.2f}% ({metrics_baseline['points_breakdown']['fairness_utility_points']:.2f})",
            "Brier Utility (10pt)": f"{metrics_baseline['brier_score']:.4f} ({metrics_baseline['points_breakdown']['calibration_utility_points']:.2f})",
            "Total Score (100pt)": f"{metrics_baseline['total_score']:.2f}"
        },
        {
            "Model": "Fair Prioritizer (Proposed)",
            "Recall @ 20% (40pt)": f"{metrics_fair['overall_recall']*100:.2f}% ({metrics_fair['points_breakdown']['recall_points']:.2f})",
            "Worst Group Rmin (25pt)": f"{metrics_fair['worst_group_recall']*100:.2f}% ({metrics_fair['points_breakdown']['worst_group_recall_points']:.2f})",
            "Fairness Gap (20pt)": f"{metrics_fair['fairness_gap']*100:.2f}% ({metrics_fair['points_breakdown']['fairness_utility_points']:.2f})",
            "Brier Utility (10pt)": f"{metrics_fair['brier_score']:.4f} ({metrics_fair['points_breakdown']['calibration_utility_points']:.2f})",
            "Total Score (100pt)": f"{metrics_fair['total_score']:.2f}"
        }
    ])
    print(comp_df.to_string(index=False))

    # 4. Save Trained Model Artifact
    fair_model.save(args.model_out)
    print(f"\nSaved production model artifact to: {args.model_out}")

    # 5. Generate Submission Deliverables on Development Data
    print("\n" + "-" * 70)
    print("Generating Submission Deliverables...")
    print("-" * 70)
    pred_results = fair_model.rank_and_select(X, budget_fraction=args.budget, ids=ids)

    # Deliverable A: Probability and ranking output
    # Columns: id, support_probability, support_rank
    prob_rank_df = pred_results[["id", "support_probability", "support_rank"]]
    prob_rank_path = os.path.join(args.output_dir, "probability_and_ranking.csv")
    prob_rank_df.to_csv(prob_rank_path, index=False)
    print(f"Deliverable 1 written: {prob_rank_path} ({len(prob_rank_df)} rows)")

    # Deliverable B: Selected-student output
    # Columns: id, support_probability, support_rank, selected_for_support
    selected_path = os.path.join(args.output_dir, "selected_students.csv")
    pred_results.to_csv(selected_path, index=False)
    print(f"Deliverable 2 written: {selected_path} ({pred_results['selected_for_support'].sum()} students selected)")

    # Deliverable C: Group-wise fairness table
    fairness_table_md = format_fairness_table(metrics_fair)
    fairness_table_path = os.path.join(args.output_dir, "fairness_table.md")
    with open(fairness_table_path, "w", encoding="utf-8") as f:
        f.write("# Group-Wise Fairness and Utility Evaluation Table\n\n")
        f.write("### Model: FairStudentPrioritizer (Calibrated Ensemble + Pareto Alignment)\n\n")
        f.write(f"- **Total Evaluated Rows**: {metrics_fair['n_total']}\n")
        f.write(f"- **Support Capacity (Top-20%)**: {metrics_fair['k_selected']} students\n")
        f.write(f"- **Overall Recall**: {metrics_fair['overall_recall'] * 100:.2f}%\n")
        f.write(f"- **Worst-Group Recall (R_min)**: {metrics_fair['worst_group_recall'] * 100:.2f}% ({metrics_fair['worst_group_name']})\n")
        f.write(f"- **Overall Fairness Gap**: {metrics_fair['fairness_gap'] * 100:.2f}%\n")
        f.write(f"- **Brier Score / Calibration Utility**: {metrics_fair['brier_score']:.4f} / {metrics_fair['calibration_utility']:.4f}\n")
        f.write(f"- **Total Judging Score**: {metrics_fair['total_score']:.2f} / 100\n\n")
        f.write("## Subgroup Performance Metrics\n\n")
        f.write(fairness_table_md)
        f.write("\n\n---\n*Criteria: A group is eligible when containing >= 10 rows and >= 3 positive examples.*\n")

    print(f"Deliverable 3 written: {fairness_table_path}")
    print("\nGroup-wise Fairness Table Summary:")
    print(fairness_table_md)

    # Save metrics JSON for automated evaluation harness
    metrics_json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    json_metrics = {k: v for k, v in metrics_fair.items() if k != "group_stats"}
    json_metrics["group_stats"] = metrics_fair["group_stats"].to_dict(orient="records")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Deliverable 4 written: {metrics_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
