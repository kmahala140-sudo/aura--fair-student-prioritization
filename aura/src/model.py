"""
Model architecture and fair prioritization pipeline for ED-01.

Features:
- Engineered academic indicators (grade velocity, critical thresholds, study-to-travel ratio)
- Leakage-free ColumnTransformer pipeline (StandardScaler + OneHotEncoder)
- Calibrated blending ensemble (L2 Logistic Regression + LightGBM)
- In-fold Pareto-optimal fair calibration optimizer for demographic parity and worst-group recall
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

from src.metrics import compute_top_k_selection, evaluate_fair_prioritization


# Columns that must NEVER be used as input features
FORBIDDEN_TARGET_COLS = ["G3", "support_needed"]
# Potential ID columns to drop from feature matrix
ID_COL_NAMES = ["id", "student_id", "eval_id", "opaque_id", "row_id", "index"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends domain-informed academic indicator features.
    
    Permitted features only: G1 and G2 are allowed as intervention happens
    after earlier assessment periods.
    """
    df_out = df.copy()

    # Grade velocity and trends
    if "G1" in df_out.columns and "G2" in df_out.columns:
        df_out["grade_delta"] = df_out["G2"] - df_out["G1"]
        df_out["grade_avg"] = (df_out["G1"] + df_out["G2"]) / 2.0
        df_out["grade_critical_g2"] = (df_out["G2"] < 10).astype(float)
        df_out["grade_critical_g1"] = (df_out["G1"] < 10).astype(float)
        df_out["grade_decay"] = (df_out["G2"] < df_out["G1"]).astype(float)

    # Academic friction and lifestyle
    if "studytime" in df_out.columns and "traveltime" in df_out.columns:
        df_out["study_to_travel"] = df_out["studytime"] / (df_out["traveltime"] + 1.0)

    if "failures" in df_out.columns:
        df_out["has_prior_failures"] = (df_out["failures"] > 0).astype(float)

    if "Dalc" in df_out.columns and "Walc" in df_out.columns:
        df_out["total_alcohol"] = df_out["Dalc"] + df_out["Walc"]

    if "absences" in df_out.columns:
        df_out["high_absences"] = (df_out["absences"] > 10).astype(float)

    return df_out


def extract_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[pd.Series]]:
    """
    Separates feature matrix X from target y and optional ID series.
    Guarantees G3 and support_needed are stripped from features.
    """
    df_clean = df.copy()

    # Extract or create ID series
    id_col = None
    for col in ID_COL_NAMES:
        if col in df_clean.columns:
            id_col = col
            break
            
    if id_col is not None:
        ids = df_clean[id_col]
        df_clean = df_clean.drop(columns=[id_col])
    else:
        ids = pd.Series(np.arange(len(df_clean)), name="student_id")

    # Extract target if present
    y = None
    if "support_needed" in df_clean.columns:
        y = df_clean["support_needed"].to_numpy(dtype=int)
        df_clean = df_clean.drop(columns=["support_needed"])
    elif "G3" in df_clean.columns:
        y = (df_clean["G3"] < 10).to_numpy(dtype=int)

    # Strictly ensure G3 is stripped
    if "G3" in df_clean.columns:
        df_clean = df_clean.drop(columns=["G3"])

    # Engineer academic features
    X = add_engineered_features(df_clean)

    return X, y, ids


class FairStudentPrioritizer(BaseEstimator, ClassifierMixin):
    """
    Calibrated Ensemble with In-Fold Fair Prioritization for ED-01.
    """

    def __init__(
        self,
        c_lr: float = 0.5,
        lgb_n_estimators: int = 50,
        lgb_max_depth: int = 3,
        lgb_lr: float = 0.04,
        ensemble_weight_lr: float = 0.50,
        enable_fair_alignment: bool = True,
        random_state: int = 42
    ):
        self.c_lr = c_lr
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_max_depth = lgb_max_depth
        self.lgb_lr = lgb_lr
        self.ensemble_weight_lr = ensemble_weight_lr
        self.enable_fair_alignment = enable_fair_alignment
        self.random_state = random_state

        # Fitted attributes
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.model_lr_: Optional[LogisticRegression] = None
        self.model_lgb_: Optional[lgb.LGBMClassifier] = None
        self.alpha_school_: float = 0.0
        self.beta_sex_: float = 0.0
        self.feature_names_: List[str] = []
        self.training_metrics_: Dict[str, Any] = {}

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "str"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols)
            ]
        )
        return preprocessor

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FairStudentPrioritizer":
        """
        Fits the ensemble models and learns fair Pareto alignment parameters
        strictly via internal stratified cross-validation without data leakage.
        """
        self.feature_names_ = list(X.columns)

        # 1. Stratified K-Fold to generate out-of-fold calibrated probabilities
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_preds_lr = np.zeros(len(X))
        oof_preds_lgb = np.zeros(len(X))

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            fold_pre = self._build_preprocessor(X_tr)
            X_tr_trans = fold_pre.fit_transform(X_tr)
            X_val_trans = fold_pre.transform(X_val)

            fold_lr = LogisticRegression(C=self.c_lr, max_iter=1000, random_state=self.random_state)
            fold_lr.fit(X_tr_trans, y_tr)
            oof_preds_lr[val_idx] = fold_lr.predict_proba(X_val_trans)[:, 1]

            fold_lgb = lgb.LGBMClassifier(
                n_estimators=self.lgb_n_estimators,
                max_depth=self.lgb_max_depth,
                learning_rate=self.lgb_lr,
                random_state=self.random_state,
                verbose=-1
            )
            fold_lgb.fit(X_tr_trans, y_tr)
            oof_preds_lgb[val_idx] = fold_lgb.predict_proba(X_val_trans)[:, 1]

        oof_blend = (
            self.ensemble_weight_lr * oof_preds_lr +
            (1.0 - self.ensemble_weight_lr) * oof_preds_lgb
        )

        # 2. Fit final production estimators on all training data
        self.preprocessor_ = self._build_preprocessor(X)
        X_trans = self.preprocessor_.fit_transform(X)

        self.model_lr_ = LogisticRegression(C=self.c_lr, max_iter=1000, random_state=self.random_state)
        self.model_lr_.fit(X_trans, y)

        self.model_lgb_ = lgb.LGBMClassifier(
            n_estimators=self.lgb_n_estimators,
            max_depth=self.lgb_max_depth,
            learning_rate=self.lgb_lr,
            random_state=self.random_state,
            verbose=-1
        )
        self.model_lgb_.fit(X_trans, y)

        # 3. Learn optimal fair Pareto alignment parameters (alpha, beta)
        if self.enable_fair_alignment and "school" in X.columns and "sex" in X.columns:
            df_oof = X.copy()
            df_oof["support_needed"] = y
            
            best_score = -1.0
            best_alpha = 0.0
            best_beta = 0.0
            best_metrics = {}

            # Search grid for fine-grained calibration shift
            alpha_grid = np.linspace(-0.10, 0.12, 45)
            beta_grid = np.linspace(-0.04, 0.04, 17)

            for a in alpha_grid:
                for b in beta_grid:
                    p_cand = oof_blend.copy()
                    p_cand += a * (df_oof["school"] == "GP").astype(float)
                    p_cand += b * (df_oof["sex"] == "F").astype(float)
                    p_cand = np.clip(p_cand, 0.0, 1.0)
                    df_oof["_cand_prob"] = p_cand

                    res = evaluate_fair_prioritization(
                        df_oof,
                        prob_col="_cand_prob",
                        target_col="support_needed",
                        budget_fraction=0.20
                    )

                    if res["total_score"] > best_score:
                        best_score = res["total_score"]
                        best_alpha = float(a)
                        best_beta = float(b)
                        best_metrics = res

            self.alpha_school_ = best_alpha
            self.beta_sex_ = best_beta
            self.training_metrics_ = best_metrics
        else:
            self.alpha_school_ = 0.0
            self.beta_sex_ = 0.0
            df_oof = X.copy()
            df_oof["support_needed"] = y
            df_oof["_raw_prob"] = oof_blend
            self.training_metrics_ = evaluate_fair_prioritization(
                df_oof,
                prob_col="_raw_prob",
                target_col="support_needed",
                budget_fraction=0.20
            )

        return self

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns unadjusted calibrated ensemble probabilities in [0, 1]."""
        if self.preprocessor_ is None or self.model_lr_ is None or self.model_lgb_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_proc = add_engineered_features(X)
        X_trans = self.preprocessor_.transform(X_proc)

        p_lr = self.model_lr_.predict_proba(X_trans)[:, 1]
        p_lgb = self.model_lgb_.predict_proba(X_trans)[:, 1]

        p_blend = self.ensemble_weight_lr * p_lr + (1.0 - self.ensemble_weight_lr) * p_lgb
        return np.clip(p_blend, 0.0, 1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns group-fair calibrated support probabilities.
        Applies learned Pareto alignment for demographic parity and worst-group recall.
        """
        p_raw = self.predict_raw_proba(X)
        if not self.enable_fair_alignment:
            return p_raw

        p_adj = p_raw.copy()
        if "school" in X.columns and self.alpha_school_ != 0.0:
            p_adj += self.alpha_school_ * (X["school"] == "GP").to_numpy(dtype=float)

        if "sex" in X.columns and self.beta_sex_ != 0.0:
            p_adj += self.beta_sex_ * (X["sex"] == "F").to_numpy(dtype=float)

        return np.clip(p_adj, 0.0, 1.0)

    def rank_and_select(
        self,
        X: pd.DataFrame,
        budget_fraction: float = 0.20,
        ids: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Ranks all students by fair probability and selects the top ceil(0.20 * N).
        
        Returns DataFrame with:
        - id
        - support_probability
        - support_rank (1 is highest priority)
        - selected_for_support (1 or 0)
        """
        n = len(X)
        probs = self.predict_proba(X)
        raw_probs = self.predict_raw_proba(X)

        if ids is None:
            ids = pd.Series(np.arange(n), name="student_id")

        # Stable descending sort: 1 is top priority
        # argsort of -probs: rank 1 is highest probability
        ranks = np.argsort(np.argsort(-probs, kind="mergesort")) + 1
        selected = compute_top_k_selection(probs, budget_fraction=budget_fraction).astype(int)

        results_df = pd.DataFrame({
            "id": ids.values,
            "support_probability": np.round(probs, 5),
            "raw_probability": np.round(raw_probs, 5),
            "support_rank": ranks,
            "selected_for_support": selected
        })

        return results_df

    def save(self, filepath: str) -> None:
        """Serializes model artifact to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FairStudentPrioritizer":
        """Loads serialized model artifact from disk."""
        return joblib.load(filepath)
