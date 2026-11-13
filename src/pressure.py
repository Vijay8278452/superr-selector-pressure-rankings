"""Pressure calculation and Leverage Index module for the SupeRR pipeline.

This module computes the win probability Leverage Index (LI) strictly from
pre-ball information. It categorises deliveries into pressure tiers, trains a
rule-based Pressure Index (PI) via non-negative least squares cross check, and
evaluates statistical differences between pressure and ordinary play.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu

from src.config import (
    LI_TIER_LOW_MAX,
    LI_TIER_MED_MAX,
    LI_TIER_HIGH_MAX,
    PRIMARY_PRESSURE_PERCENTILE,
    OUTCOME_CLASSES,
)


def compute_leverage_index(
    df: pd.DataFrame,
    wp_metrics: Dict[str, any],
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Compute pre-ball Leverage Index and pressure tiers for all deliveries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with match state, out-of-fold wp_before and outcome probabilities.
    wp_metrics : Dict[str, any]
        Dictionary holding trained win probability models and feature lists.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        Dataframe augmented with LI and pressure tiers, plus tier metrics.
    """
    df = df.copy()

    features_inn2 = wp_metrics["features_inn2"]
    features_inn1 = wp_metrics["features_inn1"]
    inn2_model = wp_metrics["models_inn2"][0]
    inn1_model = wp_metrics["models_inn1"][0]

    # Mask for legal deliveries
    mask_legal = df["is_legal"]
    mask_inn2 = df["innings"] == 2
    mask_inn1 = df["innings"] == 1

    expected_abs_swing = np.zeros(len(df), dtype=float)
    expected_wp_change = np.zeros(len(df), dtype=float)
    wp_after_actual = np.array(df["wp_before"].values, dtype=float, copy=True)

    # Pre-allocate array for outcome swings
    # Outcomes: 0, 1, 2, 3, 4, 6, wicket
    runs_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "wicket": 0}
    wkt_map = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "6": 0, "wicket": 1}

    # Store post-ball win probabilities for each outcome
    post_wp_dict: Dict[str, np.ndarray] = {}

    for c in OUTCOME_CLASSES:
        r_add = runs_map[c]
        w_add = wkt_map[c]

        post_wp = np.zeros(len(df))

        # --- Innings 2 Evaluation ---
        inn2_idx = df.index[mask_inn2].values
        if len(inn2_idx) > 0:
            sub2 = df.loc[inn2_idx].copy()
            nxt_req = np.maximum(0.0, sub2["runs_required"] - r_add)
            nxt_rem = np.maximum(0.0, sub2["balls_remaining"] - 1.0)
            nxt_wkt = np.maximum(0.0, sub2["wickets_in_hand"] - w_add)
            nxt_lgl = sub2["legal_balls_bowled"] + 1.0
            nxt_tot = sub2["team_runs"] + r_add

            # Terminal states
            is_win = nxt_req == 0.0
            is_all_out = (nxt_wkt == 0.0) & (nxt_req > 0.0)
            is_balls_up_loss = (nxt_rem == 0.0) & (nxt_req > 0.0)
            is_tie = (nxt_rem == 0.0) & (nxt_req == 0.0)

            # Non-terminal state evaluation
            non_term = ~(is_win | is_all_out | is_balls_up_loss | is_tie)

            wp_sub2 = np.zeros(len(sub2))
            wp_sub2[is_win] = 1.0
            wp_sub2[is_all_out] = 0.0
            wp_sub2[is_balls_up_loss] = 0.0
            wp_sub2[is_tie] = 0.5

            if np.any(non_term):
                sub_eval = sub2.loc[non_term].copy()
                sub_eval["runs_required"] = nxt_req[non_term]
                sub_eval["balls_remaining"] = nxt_rem[non_term]
                sub_eval["wickets_in_hand"] = nxt_wkt[non_term]
                sub_eval["legal_balls_bowled"] = nxt_lgl[non_term]
                sub_eval["team_runs"] = nxt_tot[non_term]

                sub_rem = nxt_rem[non_term] / 6.0
                sub_eval["required_run_rate"] = np.where(
                    sub_rem > 0, nxt_req[non_term] / np.maximum(0.1, sub_rem), 99.0
                )
                sub_eval["current_run_rate"] = nxt_tot[non_term] / np.maximum(0.1, nxt_lgl[non_term] / 6.0)
                sub_eval["rrr_minus_crr"] = sub_eval["required_run_rate"] - sub_eval["current_run_rate"]
                sub_eval["runs_required_per_wkt"] = nxt_req[non_term] / np.maximum(1.0, nxt_wkt[non_term])
                sub_eval["required_runs_per_ball"] = nxt_req[non_term] / np.maximum(1.0, nxt_rem[non_term])
                sub_eval["resource_proxy"] = (nxt_rem[non_term] / 120.0) * ((nxt_wkt[non_term] / 10.0) ** 0.5)

                X_eval = sub_eval[features_inn2].fillna(0).values
                wp_sub2[non_term] = inn2_model.predict_proba(X_eval)[:, 1]

            post_wp[inn2_idx] = wp_sub2

        # --- Innings 1 Evaluation ---
        inn1_idx = df.index[mask_inn1].values
        if len(inn1_idx) > 0:
            sub1 = df.loc[inn1_idx].copy()
            nxt_tot = sub1["team_runs"] + r_add
            nxt_rem = np.maximum(0.0, sub1["balls_remaining"] - 1.0)
            nxt_wkt_lost = np.minimum(10.0, sub1["wickets_lost"] + w_add)
            nxt_wkt_in_hand = np.maximum(0.0, sub1["wickets_in_hand"] - w_add)
            nxt_lgl = sub1["legal_balls_bowled"] + 1.0

            # Innings 1 ends if balls exhausted or all out
            inn_over = (nxt_rem == 0.0) | (nxt_wkt_in_hand == 0.0)
            wp_sub1 = np.zeros(len(sub1))

            if np.any(inn_over):
                # Start of Innings 2 evaluation for chasing team
                sub_over = sub1.loc[inn_over].copy()
                sub_over["runs_required"] = nxt_tot[inn_over] + 1.0
                sub_over["balls_remaining"] = 120.0
                sub_over["wickets_in_hand"] = 10.0
                sub_over["legal_balls_bowled"] = 0.0
                sub_over["team_runs"] = 0.0
                sub_over["required_run_rate"] = sub_over["runs_required"] / 20.0
                sub_over["current_run_rate"] = 8.0
                sub_over["rrr_minus_crr"] = sub_over["required_run_rate"] - 8.0
                sub_over["runs_required_per_wkt"] = sub_over["runs_required"] / 10.0
                sub_over["required_runs_per_ball"] = sub_over["runs_required"] / 120.0
                sub_over["resource_proxy"] = 1.0

                X_over = sub_over[features_inn2].fillna(0).values
                chasing_win = inn2_model.predict_proba(X_over)[:, 1]
                wp_sub1[inn_over] = 1.0 - chasing_win

            if np.any(~inn_over):
                sub_eval = sub1.loc[~inn_over].copy()
                sub_eval["team_runs"] = nxt_tot[~inn_over]
                sub_eval["balls_remaining"] = nxt_rem[~inn_over]
                sub_eval["wickets_lost"] = nxt_wkt_lost[~inn_over]
                sub_eval["wickets_in_hand"] = nxt_wkt_in_hand[~inn_over]
                sub_eval["legal_balls_bowled"] = nxt_lgl[~inn_over]

                sub_rem = nxt_rem[~inn_over] / 6.0
                sub_eval["current_run_rate"] = nxt_tot[~inn_over] / np.maximum(0.1, nxt_lgl[~inn_over] / 6.0)
                sub_eval["projected_score"] = (
                    nxt_tot[~inn_over]
                    + sub_rem * sub_eval["current_run_rate"] * ((nxt_wkt_in_hand[~inn_over] / 10.0) ** 0.3)
                )
                sub_eval["run_rate_shortfall"] = (
                    (sub_eval["venue_par_score"] - nxt_tot[~inn_over]) / np.maximum(1.0, sub_rem)
                    - sub_eval["current_run_rate"]
                )

                X_eval = sub_eval[features_inn1].fillna(0).values
                wp_sub1[~inn_over] = inn1_model.predict_proba(X_eval)[:, 1]

            post_wp[inn1_idx] = wp_sub1

        post_wp_dict[c] = np.clip(post_wp, 0.0, 1.0)

        # Swing calculation
        swing = np.abs(post_wp_dict[c] - df["wp_before"].values)
        prob_c = df[f"prob_{c}"].values
        expected_abs_swing += prob_c * swing
        expected_wp_change += prob_c * (post_wp_dict[c] - df["wp_before"].values)

    # Actual outcome after delivery
    actual_outcome = df["outcome_label"].values
    for c in OUTCOME_CLASSES:
        match_c = ((actual_outcome == c) & mask_legal).values
        wp_after_actual[match_c] = post_wp_dict[c][match_c]

    # For illegal deliveries, post-ball WP changes based on runs conceded
    mask_illegal = (~mask_legal).values
    if np.any(mask_illegal):
        wp_after_actual[mask_illegal] = np.clip(
            df.loc[mask_illegal, "wp_before"].values + df.loc[mask_illegal, "total_ball_runs"].values * 0.012,
            0.0,
            1.0,
        )

    # Scale Leverage Index so mean across all legal deliveries equals 1.0
    legal_mean_swing = float(expected_abs_swing[mask_legal].mean())
    if legal_mean_swing <= 0:
        legal_mean_swing = 1.0

    df["leverage_index"] = expected_abs_swing / legal_mean_swing
    # For non-legal balls, assign the state LI
    df.loc[mask_illegal, "leverage_index"] = df.loc[mask_illegal, "leverage_index"].fillna(1.0)

    # Pre-ball Pressure Tiers
    legal_li = df.loc[mask_legal, "leverage_index"]
    cut_low = float(legal_li.quantile(LI_TIER_LOW_MAX))
    cut_med = float(legal_li.quantile(LI_TIER_MED_MAX))
    cut_high = float(legal_li.quantile(LI_TIER_HIGH_MAX))

    tier_conditions = [
        df["leverage_index"] < cut_low,
        (df["leverage_index"] >= cut_low) & (df["leverage_index"] < cut_med),
        (df["leverage_index"] >= cut_med) & (df["leverage_index"] < cut_high),
        df["leverage_index"] >= cut_high,
    ]
    tier_choices = ["Low", "Medium", "High", "Extreme"]
    df["pressure_tier"] = np.select(tier_conditions, tier_choices, default="Low")

    # Primary pressure flag: High plus Extreme (top 25 percent)
    df["is_pressure"] = df["pressure_tier"].isin(["High", "Extreme"])

    # Rule-based Pressure Index (PI) cross check
    pi_features = pd.DataFrame(index=df.index)
    pi_features["rrr_pressure"] = np.where(
        df["innings"] == 2,
        (df["required_run_rate"] / 16.0).clip(0.0, 1.0),
        (df["run_rate_shortfall"] / 5.0).clip(0.0, 1.0),
    )
    pi_features["wicket_pressure"] = ((10.0 - df["wickets_in_hand"]) / 10.0) ** 1.4
    pi_features["time_pressure"] = (120.0 - df["balls_remaining"]) / 120.0
    pi_features["closeness"] = 1.0 - 2.0 * np.abs(df["wp_before"] - 0.5)
    pi_features["momentum"] = (df["balls_since_last_boundary"] / 20.0).clip(0.0, 1.0)

    X_pi = pi_features.fillna(0.0).values
    y_li = df["leverage_index"].values

    weights, _ = nnls(X_pi, y_li)
    fitted_pi = X_pi @ weights
    df["pressure_index_rule"] = fitted_pi

    # Metrics dictionary
    corr_pi_li = float(np.corrcoef(fitted_pi, y_li)[0, 1])
    ss_tot = np.sum((y_li - np.mean(y_li)) ** 2)
    ss_res = np.sum((y_li - fitted_pi) ** 2)
    r2_pi = float(1.0 - ss_res / ss_tot)

    df["wp_after"] = wp_after_actual
    df["expected_wp_change"] = expected_wp_change
    df["actual_wp_change"] = wp_after_actual - df["wp_before"].values

    metrics = {
        "cut_low": cut_low,
        "cut_med": cut_med,
        "cut_high": cut_high,
        "pi_weights": weights.tolist(),
        "pi_feature_names": ["Required Rate", "Wickets Lost", "Time (Balls)", "Closeness", "Momentum"],
        "pi_r2": r2_pi,
        "pi_correlation": corr_pi_li,
        "post_wp_dict": post_wp_dict,
    }

    return df, metrics


def compute_tier_outcome_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Compute league-wide batting and bowling outcomes across pressure tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with leverage index and pressure tiers.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, any]]
        Table 3 dataframe and statistical comparison test results.
    """
    records = []
    tiers = ["Low", "Medium", "High", "Extreme"]

    for tier in tiers:
        sub = df[df["pressure_tier"] == tier]
        legal_sub = sub[sub["is_legal"]]

        n_balls = len(sub)
        n_legal = len(legal_sub)
        runs_bat = legal_sub["bat_runs"].sum()
        dots = (legal_sub["bat_runs"] == 0) & (~legal_sub["is_wicket"])
        boundaries = legal_sub["is_four"] | legal_sub["is_six"]
        wickets = legal_sub["is_wicket"].sum()
        runs_bowler = sub["bowler_runs_conceded"].sum()
        extras = sub["extras_type"].isin(["wide", "Overstep", "other"]).sum()

        strike_rate = (runs_bat / n_legal) * 100.0 if n_legal > 0 else 0.0
        dot_pct = (dots.sum() / n_legal) * 100.0 if n_legal > 0 else 0.0
        boundary_pct = (boundaries.sum() / n_legal) * 100.0 if n_legal > 0 else 0.0
        dismissal_rate = (wickets / n_legal) * 100.0 if n_legal > 0 else 0.0
        economy = (runs_bowler / n_legal) * 6.0 if n_legal > 0 else 0.0
        extras_rate = (extras / n_balls) * 100.0 if n_balls > 0 else 0.0

        records.append({
            "Tier": tier,
            "Deliveries": n_balls,
            "Legal Balls": n_legal,
            "Strike Rate": round(strike_rate, 2),
            "Dot Ball Pct": round(dot_pct, 2),
            "Boundary Pct": round(boundary_pct, 2),
            "Dismissal Pct": round(dismissal_rate, 2),
            "Economy Rate": round(economy, 2),
            "Extras Pct": round(extras_rate, 2),
        })

    tier_df = pd.DataFrame(records)

    # Statistical test between Pressure (High + Extreme) and Ordinary (Low + Medium)
    pressure_legal = df[df["is_pressure"] & df["is_legal"]]
    ordinary_legal = df[(~df["is_pressure"]) & df["is_legal"]]

    # Mann-Whitney U test on runs scored per ball
    u_stat, mw_pvalue = mannwhitneyu(
        pressure_legal["bat_runs"].values,
        ordinary_legal["bat_runs"].values,
        alternative="two-sided",
    )

    # Dismissal rate two-proportion comparison
    p1 = pressure_legal["is_wicket"].mean()
    p2 = ordinary_legal["is_wicket"].mean()
    n1 = len(pressure_legal)
    n2 = len(ordinary_legal)
    p_pool = (pressure_legal["is_wicket"].sum() + ordinary_legal["is_wicket"].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z_stat = (p1 - p2) / se

    tests = {
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(mw_pvalue),
        "dismissal_z": float(z_stat),
        "dismissal_diff_pct": float((p1 - p2) * 100.0),
    }

    return tier_df, tests
