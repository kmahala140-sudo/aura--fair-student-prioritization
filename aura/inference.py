"""
Inference script for ED-01: Fair Student-Support Prioritization.

Designed for evaluator execution on held-out evaluation datasets.

Usage:
    python inference.py --input <path_to_input_csv> \
                        --output_probabilities outputs/probability_and_ranking.csv \
                        --output_selected outputs/selected_students.csv \
                        --model models/fair_student_model.pkl

Constraints Satisfied:
- Permitted feature fields only (G1, G2, demographics, school context).
- No dependency on G3 or ground-truth support_needed targets.
- Opaque evaluation ID preservation.
- Zero external lookups, fingerprinting, or target recovery.
- Capacity-constrained selection: exactly ceil(0.20 * N) students selected.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from src.model import FairStudentPrioritizer, ID_COL_NAMES, add_engineered_features


def read_evaluation_csv(filepath: str) -> pd.DataFrame:
    """
    Reads evaluation CSV with automatic delimiter detection (; or ,).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at: {filepath}")

    # Try semicolon first
    try:
        df = pd.read_csv(filepath, sep=";")
        if df.shape[1] > 5:
            return df
    except Exception:
        pass

    # Fallback to comma
    return pd.read_csv(filepath, sep=",")


def extract_evaluation_features_and_id(
    df: pd.DataFrame,
    specified_id_col: str = None
):
    """
    Extracts opaque ID series and sanitizes feature dataframe.
    Guarantees no target or evaluator-only metadata is passed to the model.
    """
    df_eval = df.copy()

    # Find ID column
    id_col = None
    if specified_id_col and specified_id_col in df_eval.columns:
        id_col = specified_id_col
    else:
        for candidate in ID_COL_NAMES:
            if candidate in df_eval.columns:
                id_col = candidate
                break

    if id_col is not None:
        ids = df_eval[id_col]
        X = df_eval.drop(columns=[id_col])
    else:
        ids = pd.Series(np.arange(len(df_eval)), name="id")
        X = df_eval

    # Ensure any target fields are strictly dropped if present
    for target in ["G3", "support_needed"]:
        if target in X.columns:
            X = X.drop(columns=[target])

    return X, ids


def main():
    parser = argparse.ArgumentParser(description="Inference for Fair Student-Support Prioritization.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/student-mat.csv",
        help="Path to evaluation CSV file containing student feature rows."
    )
    parser.add_argument(
        "--output_probabilities", "-p",
        type=str,
        default="outputs/probability_and_ranking.csv",
        help="Path for probability and ranking output CSV."
    )
    parser.add_argument(
        "--output_selected", "-s",
        type=str,
        default="outputs/selected_students.csv",
        help="Path for selected-student output CSV."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/fair_student_model.pkl",
        help="Path to serialized model artifact."
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default=None,
        help="Name of the ID column (auto-detected if omitted)."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.20,
        help="Support selection capacity fraction (default 0.20)."
    )

    args = parser.parse_args()

    # Ensure output directories exist
    if args.output_probabilities:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_probabilities)), exist_ok=True)
    if args.output_selected:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_selected)), exist_ok=True)

    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model artifact not found at {args.model}. Run 'python train.py' first."
        )
    model = FairStudentPrioritizer.load(args.model)

    print(f"Reading evaluation data from: {args.input}")
    df_raw = read_evaluation_csv(args.input)
    n_rows = len(df_raw)
    print(f"Evaluated row count: {n_rows}")

    X, ids = extract_evaluation_features_and_id(df_raw, args.id_col)
    k_expected = int(np.ceil(args.budget * n_rows))
    print(f"Support budget (ceil({args.budget:.2f} * {n_rows})): exactly {k_expected} students")

    # Run ranking and selection
    results_df = model.rank_and_select(X, budget_fraction=args.budget, ids=ids)

    # Validate output constraints
    actual_selected = int(results_df["selected_for_support"].sum())
    assert actual_selected == k_expected, (
        f"Selection count mismatch: expected {k_expected}, got {actual_selected}"
    )

    # Deliverable 1: Probability and ranking output
    # Columns: id, support_probability, support_rank
    # Sorted by support_rank ascending (rank 1 is highest priority)
    prob_rank_df = (
        results_df[["id", "support_probability", "support_rank"]]
        .sort_values(by="support_rank", ascending=True)
        .reset_index(drop=True)
    )
    prob_rank_df.to_csv(args.output_probabilities, index=False)
    print(f"Wrote probability and ranking output to: {args.output_probabilities}")

    # Deliverable 2: Selected-student output
    # Columns: id, support_probability, support_rank, selected_for_support
    selected_df = (
        results_df[["id", "support_probability", "support_rank", "selected_for_support"]]
        .sort_values(by="support_rank", ascending=True)
        .reset_index(drop=True)
    )
    selected_df.to_csv(args.output_selected, index=False)
    print(f"Wrote selected-student output to: {args.output_selected}")

    # Summary breakdown
    print("-" * 60)
    print("Inference Selection Summary:")
    print(f"  - Total Students Evaluated: {n_rows}")
    print(f"  - Students Selected: {actual_selected} ({actual_selected / n_rows * 100:.1f}%)")
    if "school" in df_raw.columns:
        school_counts = df_raw.groupby(df_raw["school"]).size()
        sel_schools = df_raw.loc[results_df["selected_for_support"] == 1, "school"].value_counts()
        print("  - Selection by School:")
        for s, count in school_counts.items():
            sel_s = sel_schools.get(s, 0)
            print(f"    * {s}: {sel_s} / {count} ({sel_s / count * 100:.1f}%)")

    if "sex" in df_raw.columns:
        sex_counts = df_raw.groupby(df_raw["sex"]).size()
        sel_sex = df_raw.loc[results_df["selected_for_support"] == 1, "sex"].value_counts()
        print("  - Selection by Sex:")
        for s, count in sex_counts.items():
            sel_s = sel_sex.get(s, 0)
            print(f"    * {s}: {sel_s} / {count} ({sel_s / count * 100:.1f}%)")
    print("-" * 60)
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
