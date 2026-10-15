"""
End-to-end integration tests for inference.py.
"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest


def test_inference_cli_execution(tmp_path):
    # Create minimal test dataset with opaque student IDs
    n = 25
    df_test = pd.DataFrame({
        "opaque_id": [f"TEST_{i}" for i in range(n)],
        "school": ["GP"] * 20 + ["MS"] * 5,
        "sex": ["F"] * 13 + ["M"] * 12,
        "age": [16] * n,
        "address": ["U"] * n,
        "famsize": ["GT3"] * n,
        "Pstatus": ["T"] * n,
        "Medu": [3] * n,
        "Fedu": [3] * n,
        "Mjob": ["other"] * n,
        "Fjob": ["other"] * n,
        "reason": ["home"] * n,
        "guardian": ["mother"] * n,
        "traveltime": [1] * n,
        "studytime": [2] * n,
        "failures": [0] * n,
        "schoolsup": ["no"] * n,
        "famsup": ["no"] * n,
        "paid": ["no"] * n,
        "activities": ["no"] * n,
        "nursery": ["yes"] * n,
        "higher": ["yes"] * n,
        "internet": ["yes"] * n,
        "romantic": ["no"] * n,
        "famrel": [4] * n,
        "freetime": [3] * n,
        "goout": [3] * n,
        "Dalc": [1] * n,
        "Walc": [1] * n,
        "health": [3] * n,
        "absences": [2] * n,
        "G1": np.random.randint(5, 15, size=n),
        "G2": np.random.randint(5, 15, size=n)
    })
    
    test_csv = tmp_path / "test_eval.csv"
    out_prob = tmp_path / "prob_rank.csv"
    out_sel = tmp_path / "selected.csv"
    df_test.to_csv(test_csv, index=False)

    cmd = [
        sys.executable,
        "inference.py",
        "--input", str(test_csv),
        "--output_probabilities", str(out_prob),
        "--output_selected", str(out_sel),
        "--model", "models/fair_student_model.pkl"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference command failed with: {res.stderr}"

    assert os.path.exists(out_prob)
    assert os.path.exists(out_sel)

    df_sel = pd.read_csv(out_sel)
    # ceil(0.20 * 25) = 5
    assert df_sel["selected_for_support"].sum() == 5
    assert list(df_sel.columns) == ["id", "support_probability", "support_rank", "selected_for_support"]
    assert df_sel.loc[0, "support_rank"] == 1
