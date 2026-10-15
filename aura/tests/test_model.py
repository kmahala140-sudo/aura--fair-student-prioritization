"""
Unit tests for FairStudentPrioritizer model.
"""

import os
import numpy as np
import pandas as pd
import pytest
from src.model import FairStudentPrioritizer, add_engineered_features, extract_features_and_target


def test_add_engineered_features():
    df = pd.DataFrame({
        "G1": [5, 12],
        "G2": [6, 14],
        "studytime": [2, 4],
        "traveltime": [1, 2],
        "failures": [0, 1],
        "Dalc": [1, 2],
        "Walc": [1, 3],
        "absences": [2, 12]
    })
    df_eng = add_engineered_features(df)
    assert "grade_delta" in df_eng.columns
    assert "grade_avg" in df_eng.columns
    assert "study_to_travel" in df_eng.columns
    assert "high_absences" in df_eng.columns
    assert df_eng.loc[0, "grade_delta"] == 1.0
    assert df_eng.loc[1, "grade_delta"] == 2.0


def test_extract_features_and_target():
    df = pd.DataFrame({
        "id": [101, 102],
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "G1": [8, 14],
        "G2": [7, 15],
        "G3": [6, 16]
    })
    X, y, ids = extract_features_and_target(df)
    assert "G3" not in X.columns
    assert "id" not in X.columns
    assert len(y) == 2
    assert y[0] == 1  # G3 < 10 -> support_needed = 1
    assert y[1] == 0
    assert list(ids) == [101, 102]


def test_model_fit_predict_save_load(tmp_path):
    # Create small synthetic dataset
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "school": np.random.choice(["GP", "MS"], size=n),
        "sex": np.random.choice(["F", "M"], size=n),
        "G1": np.random.randint(4, 18, size=n),
        "G2": np.random.randint(4, 18, size=n),
        "studytime": np.random.randint(1, 5, size=n),
        "traveltime": np.random.randint(1, 4, size=n),
        "failures": np.random.randint(0, 3, size=n),
        "absences": np.random.randint(0, 20, size=n)
    })
    y = ((df["G2"] < 10) | (df["failures"] > 0)).astype(int).to_numpy()

    model = FairStudentPrioritizer(
        c_lr=1.0,
        lgb_n_estimators=10,
        lgb_max_depth=2,
        random_state=42
    )
    model.fit(df, y)

    probs = model.predict_proba(df)
    assert len(probs) == n
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test rank and select
    res = model.rank_and_select(df, budget_fraction=0.20)
    expected_selected = int(np.ceil(0.20 * n))
    assert res["selected_for_support"].sum() == expected_selected

    # Test serialization
    save_file = str(tmp_path / "test_model.pkl")
    model.save(save_file)
    assert os.path.exists(save_file)

    loaded = FairStudentPrioritizer.load(save_file)
    probs_loaded = loaded.predict_proba(df)
    np.testing.assert_allclose(probs, probs_loaded, rtol=1e-5)
