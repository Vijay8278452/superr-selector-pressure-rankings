"""Ball outcome multiclass modelling module for the SupeRR pressure analysis pipeline.

This module fits a context-only multiclass model over legal ball outcomes:
0, 1, 2, 3, 4, 6 and wicket. It uses out-of-fold GroupKFold cross validation
without player identifiers to provide expected values for an average player.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from src.config import RANDOM_SEED, CV_FOLDS, OUTCOME_MODEL_PARAMS, OUTCOME_CLASSES


def train_ball_outcome_model(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Train out-of-fold multiclass outcome model on legal deliveries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with match state features and wp_before.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        Dataframe with predicted outcome probabilities and expected runs,
        and evaluation metrics dictionary.
    """
    df = df.copy()

    # Define legal ball outcome categories: 0, 1, 2, 3, 4 (includes rare 5), 6, wicket
    conditions = [
        df["is_wicket"] & df["is_legal"],
        (df["bat_runs"] == 0) & (~df["is_wicket"]) & df["is_legal"],
        (df["bat_runs"] == 1) & (~df["is_wicket"]) & df["is_legal"],
        (df["bat_runs"] == 2) & (~df["is_wicket"]) & df["is_legal"],
        (df["bat_runs"] == 3) & (~df["is_wicket"]) & df["is_legal"],
        (df["bat_runs"].isin([4, 5])) & (~df["is_wicket"]) & df["is_legal"],
        (df["bat_runs"] == 6) & (~df["is_wicket"]) & df["is_legal"],
    ]
    class_names = ["wicket", "0", "1", "2", "3", "4", "6"]
    df["outcome_label"] = np.select(conditions, class_names, default="none")

    class_to_idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
    mask_legal = df["is_legal"] & df["outcome_label"].isin(class_to_idx.keys())

    legal_df = df[mask_legal].copy()
    y_outcome = legal_df["outcome_label"].map(class_to_idx).values

    # Context features only (strictly pre-ball, no player identifiers)
    features = [
        "legal_balls_bowled",
        "balls_remaining",
        "team_runs",
        "wickets_lost",
        "wickets_in_hand",
        "current_run_rate",
        "recent_run_rate",
        "balls_since_last_boundary",
        "balls_since_last_wicket",
        "partnership_runs",
        "batting_order",
        "is_rhb",
        "is_pace",
        "is_ipl",
        "elo_diff",
        "venue_par_score",
    ]

    X = legal_df[features].fillna(0).values
    groups = legal_df["match_id"].values

    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof_probs = np.zeros((len(legal_df), len(OUTCOME_CLASSES)))
    outcome_models = []

    for trn, val in gkf.split(X, y_outcome, groups):
        clf = HistGradientBoostingClassifier(
            **OUTCOME_MODEL_PARAMS,
        )
        clf.fit(X[trn], y_outcome[trn])
        oof_probs[val] = clf.predict_proba(X[val])
        outcome_models.append(clf)

    # Compute naive phase-only baseline
    phase_priors = pd.crosstab(legal_df["phase"], legal_df["outcome_label"], normalize="index")
    naive_probs = legal_df["phase"].map(
        lambda p: [phase_priors.loc[p, c] if c in phase_priors.columns else 0.0 for c in OUTCOME_CLASSES]
    ).tolist()
    naive_probs = np.array(naive_probs)

    model_logloss = float(log_loss(y_outcome, oof_probs))
    naive_logloss = float(log_loss(y_outcome, naive_probs))

    # Assign probabilities back to main dataframe
    for i, c in enumerate(OUTCOME_CLASSES):
        col_name = f"prob_{c}"
        df[col_name] = 0.0
        df.loc[mask_legal, col_name] = oof_probs[:, i]

    # Expected bat runs: 0*p(0) + 1*p(1) + 2*p(2) + 3*p(3) + 4*p(4) + 6*p(6)
    df["expected_runs"] = (
        1.0 * df["prob_1"]
        + 2.0 * df["prob_2"]
        + 3.0 * df["prob_3"]
        + 4.0 * df["prob_4"]
        + 6.0 * df["prob_6"]
    )
    df["expected_wicket"] = df["prob_wicket"]

    # Situational bowling extras baseline (wide and no-ball probabilities by phase)
    extras_rates = df.groupby("phase").agg(
        wide_rate=("extras_type", lambda s: (s == "wide").mean()),
        noball_rate=("extras_type", lambda s: s.isin(["Overstep", "other"]).mean()),
    )
    df["prob_wide"] = df["phase"].map(extras_rates["wide_rate"]).fillna(0.038)
    df["prob_noball"] = df["phase"].map(extras_rates["noball_rate"]).fillna(0.004)

    metrics = {
        "outcome_model_logloss": model_logloss,
        "naive_phase_logloss": naive_logloss,
        "outcome_features": features,
        "class_names": list(OUTCOME_CLASSES),
        "outcome_models": outcome_models,
    }

    return df, metrics
