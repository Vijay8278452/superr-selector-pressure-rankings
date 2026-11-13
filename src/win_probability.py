"""Win probability modelling module for the SupeRR pressure analysis pipeline.

This module fits calibrated out-of-fold win probability models for innings 1 and
innings 2 using GroupKFold cross validation grouped by match. Monotonic
constraints enforce cricket logic so that wickets and runs required decrease
the chasing team's probability of victory.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from src.config import RANDOM_SEED, CV_FOLDS, WP_MODEL_PARAMS


def train_win_probability_models(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Train out-of-fold win probability models for innings 1 and innings 2.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with match state features.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        Dataframe with out-of-fold win probability column 'wp_before',
        and a metrics dictionary for evaluation reporting.
    """
    df = df.copy()

    # Features for Innings 2 (chases)
    features_inn2 = [
        "runs_required",
        "balls_remaining",
        "wickets_in_hand",
        "legal_balls_bowled",
        "team_runs",
        "required_run_rate",
        "rrr_minus_crr",
        "runs_required_per_wkt",
        "required_runs_per_ball",
        "resource_proxy",
        "current_run_rate",
        "recent_run_rate",
        "balls_since_last_boundary",
        "balls_since_last_wicket",
        "partnership_runs",
        "elo_diff",
        "is_ipl",
        "is_knockout",
    ]
    # Monotonic constraints: -1 for decreasing, +1 for increasing, 0 for unconstrained
    mono_inn2 = [-1, 1, 1, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 1, 0, 0]

    # Features for Innings 1
    features_inn1 = [
        "team_runs",
        "legal_balls_bowled",
        "balls_remaining",
        "wickets_lost",
        "wickets_in_hand",
        "current_run_rate",
        "recent_run_rate",
        "projected_score",
        "venue_par_score",
        "run_rate_shortfall",
        "balls_since_last_boundary",
        "balls_since_last_wicket",
        "partnership_runs",
        "elo_diff",
        "is_ipl",
        "is_knockout",
    ]
    mono_inn1 = [1, 0, 0, -1, 1, 1, 0, 1, -1, -1, 0, 0, 0, 1, 0, 0]

    # 1. Train Innings 2 Out-Of-Fold
    mask_inn2 = df["innings"] == 2
    X2 = df.loc[mask_inn2, features_inn2].fillna(0).values
    y2 = df.loc[mask_inn2, "batting_side_win"].values
    groups2 = df.loc[mask_inn2, "match_id"].values

    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof2_wp = np.zeros(len(X2))
    oof2_lr = np.zeros(len(X2))
    inn2_models = []

    for trn, val in gkf.split(X2, y2, groups2):
        clf = HistGradientBoostingClassifier(
            monotonic_cst=mono_inn2,
            **WP_MODEL_PARAMS,
        )
        clf.fit(X2[trn], y2[trn])
        oof2_wp[val] = clf.predict_proba(X2[val])[:, 1]
        inn2_models.append(clf)

        # Baseline Logistic Regression
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
        lr.fit(scaler.fit_transform(X2[trn]), y2[trn])
        oof2_lr[val] = lr.predict_proba(scaler.transform(X2[val]))[:, 1]

    # 2. Train Innings 1 Out-Of-Fold
    mask_inn1 = df["innings"] == 1
    X1 = df.loc[mask_inn1, features_inn1].fillna(0).values
    y1 = df.loc[mask_inn1, "batting_side_win"].values
    groups1 = df.loc[mask_inn1, "match_id"].values

    oof1_wp = np.zeros(len(X1))
    oof1_lr = np.zeros(len(X1))
    inn1_models = []

    for trn, val in gkf.split(X1, y1, groups1):
        clf = HistGradientBoostingClassifier(
            monotonic_cst=mono_inn1,
            **WP_MODEL_PARAMS,
        )
        clf.fit(X1[trn], y1[trn])
        oof1_wp[val] = clf.predict_proba(X1[val])[:, 1]
        inn1_models.append(clf)

        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
        lr.fit(scaler.fit_transform(X1[trn]), y1[trn])
        oof1_lr[val] = lr.predict_proba(scaler.transform(X1[val]))[:, 1]

    # Assign out-of-fold win probability
    df["wp_before"] = 0.5
    df.loc[mask_inn2, "wp_before"] = np.clip(oof2_wp, 0.001, 0.999)
    df.loc[mask_inn1, "wp_before"] = np.clip(oof1_wp, 0.001, 0.999)

    # Compute evaluation metrics
    metrics = {
        "inn2_hgb_logloss": float(log_loss(y2, oof2_wp)),
        "inn2_hgb_auc": float(roc_auc_score(y2, oof2_wp)),
        "inn2_hgb_brier": float(brier_score_loss(y2, oof2_wp)),
        "inn2_lr_logloss": float(log_loss(y2, oof2_lr)),
        "inn2_lr_auc": float(roc_auc_score(y2, oof2_lr)),
        "inn2_lr_brier": float(brier_score_loss(y2, oof2_lr)),
        "inn1_hgb_logloss": float(log_loss(y1, oof1_wp)),
        "inn1_hgb_auc": float(roc_auc_score(y1, oof1_wp)),
        "inn1_hgb_brier": float(brier_score_loss(y1, oof1_wp)),
        "inn1_lr_logloss": float(log_loss(y1, oof1_lr)),
        "inn1_lr_auc": float(roc_auc_score(y1, oof1_lr)),
        "inn1_lr_brier": float(brier_score_loss(y1, oof1_lr)),
        "features_inn1": features_inn1,
        "features_inn2": features_inn2,
        "models_inn1": inn1_models,
        "models_inn2": inn2_models,
    }

    # Calibration curves for plotting
    prob_true_2, prob_pred_2 = calibration_curve(y2, oof2_wp, n_bins=10)
    prob_true_1, prob_pred_1 = calibration_curve(y1, oof1_wp, n_bins=10)
    metrics["calib_inn2"] = (prob_true_2, prob_pred_2)
    metrics["calib_inn1"] = (prob_true_1, prob_pred_1)

    return df, metrics
