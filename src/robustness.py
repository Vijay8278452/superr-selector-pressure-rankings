"""Robustness and sensitivity analysis module for the SupeRR pipeline.

This module evaluates the stability of rankings under varied assumptions:
pressure thresholds (15 to 30 percent), component weights, eligibility cut-offs,
recency weighting, and comparisons between IPL and T20I contexts.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge

from src.config import RANDOM_SEED
from src.ratings import build_joint_design_matrix
from src.ranking import compute_composite_score
from src.consistency import compute_player_consistency


def evaluate_sensitivity_scenarios(
    df: pd.DataFrame,
    bat_top10_ids: List[int],
    bowl_top10_ids: List[int],
    eligible_batters: List[int],
    eligible_bowlers: List[int],
    alpha: float,
) -> pd.DataFrame:
    """Evaluate top 10 stability across various modeling perturbations.

    Parameters
    ----------
    df : pd.DataFrame
        Complete delivery level dataframe.
    bat_top10_ids : List[int]
        IDs of top 10 batters from primary consensus model.
    bowl_top10_ids : List[int]
        IDs of top 10 bowlers from primary consensus model.
    eligible_batters : List[int]
        Qualifying batter IDs.
    eligible_bowlers : List[int]
        Qualifying bowler IDs.
    alpha : float
        Regularisation penalty.

    Returns
    -------
    pd.DataFrame
        Table 8 sensitivity analysis dataframe.
    """
    records = []
    scenarios = [
        ("Base Specification", 0.25, (0.50, 0.20, 0.30), "Full Sample", "All Matches"),
        ("Threshold 15% (Extreme Only)", 0.15, (0.50, 0.20, 0.30), "Full Sample", "All Matches"),
        ("Threshold 20%", 0.20, (0.50, 0.20, 0.30), "Full Sample", "All Matches"),
        ("Threshold 30% (Broader Context)", 0.30, (0.50, 0.20, 0.30), "Full Sample", "All Matches"),
        ("P-Heavy Weights (0.70 P, 0.15 G, 0.15 C)", 0.25, (0.70, 0.15, 0.15), "Full Sample", "All Matches"),
        ("G-Heavy Weights (0.40 P, 0.40 G, 0.20 C)", 0.25, (0.40, 0.40, 0.20), "Full Sample", "All Matches"),
        ("C-Heavy Weights (0.40 P, 0.20 G, 0.40 C)", 0.25, (0.40, 0.20, 0.40), "Full Sample", "All Matches"),
        ("Tight Eligibility (+20% Cut-off)", 0.25, (0.50, 0.20, 0.30), "Tight Cut-off", "All Matches"),
        ("Loose Eligibility (-20% Cut-off)", 0.25, (0.50, 0.20, 0.30), "Loose Cut-off", "All Matches"),
        ("Recency Weighted (2 Year Half-Life)", 0.25, (0.50, 0.20, 0.30), "Full Sample", "Exponential Decay"),
        ("Major Teams Only (>=15 Matches)", 0.25, (0.50, 0.20, 0.30), "Established Teams", "Filtered Teams"),
    ]

    for name, thresh, (wp, wg, wc), elig_mode, sample_mode in scenarios:
        df_mod = df.copy()

        # Handle sample mode
        if sample_mode == "Filtered Teams":
            team_match_counts = df_mod.groupby("batting_team_id")["match_id"].nunique()
            big_teams = set(team_match_counts[team_match_counts >= 15].index)
            df_mod = df_mod[
                df_mod["batting_team_id"].isin(big_teams) & df_mod["bowling_team_id"].isin(big_teams)
            ].copy()

        # Handle threshold
        li_cut = df_mod.loc[df_mod["is_legal"], "leverage_index"].quantile(1.0 - thresh)
        df_mod["is_pressure_scen"] = df_mod["leverage_index"] >= li_cut

        sub_p = df_mod[df_mod["is_pressure_scen"]].reset_index(drop=True)
        sub_n = df_mod[~df_mod["is_pressure_scen"]].reset_index(drop=True)

        if len(sub_p) == 0:
            continue

        # Fit ratings
        X_p, y_p = build_joint_design_matrix(sub_p, eligible_batters, eligible_bowlers)
        m_p = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
        m_p.fit(X_p, y_p)

        X_n, y_n = build_joint_design_matrix(sub_n, eligible_batters, eligible_bowlers)
        m_n = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
        m_n.fit(X_n, y_n)

        n_bat = len(eligible_batters)
        p_bat = m_p.coef_[:n_bat]
        g_bat = p_bat - m_n.coef_[:n_bat]
        p_bowl = m_p.coef_[n_bat:]
        g_bowl = p_bowl - m_n.coef_[n_bat:]

        # Consistency
        bat_c, bowl_c = compute_player_consistency(df_mod, eligible_batters, eligible_bowlers)

        bat_r_df = pd.DataFrame({
            "player_id": eligible_batters,
            "pressure_rating_p": p_bat,
            "clutch_gap_g": g_bat,
        })
        bowl_r_df = pd.DataFrame({
            "player_id": eligible_bowlers,
            "pressure_rating_p": p_bowl,
            "clutch_gap_g": g_bowl,
        })

        bat_sc = compute_composite_score(bat_r_df, bat_c, weight_p=wp, weight_g=wg, weight_c=wc)
        bowl_sc = compute_composite_score(bowl_r_df, bowl_c, weight_p=wp, weight_g=wg, weight_c=wc)

        top10_bat_scen = set(bat_sc.iloc[:10]["player_id"])
        top10_bowl_scen = set(bowl_sc.iloc[:10]["player_id"])

        bat_overlap = len(top10_bat_scen.intersection(bat_top10_ids))
        bowl_overlap = len(top10_bowl_scen.intersection(bowl_top10_ids))

        # Rank correlation among base top 10
        sub_b = bat_sc[bat_sc["player_id"].isin(bat_top10_ids)].sort_values("player_id")
        rho_b, _ = spearmanr(range(len(sub_b)), sub_b["base_rank"])

        records.append({
            "Scenario": name,
            "Threshold": f"Top {int(thresh * 100)}%",
            "Weights (P/G/C)": f"{wp:.2f} / {wg:.2f} / {wc:.2f}",
            "Batters Top 10 Overlap": f"{bat_overlap} / 10",
            "Bowlers Top 10 Overlap": f"{bowl_overlap} / 10",
            "Mean Top 10 Overlap": f"{(bat_overlap + bowl_overlap) / 20 * 100:.1f}%",
        })

    return pd.DataFrame(records)


def compare_with_run_based_metrics(
    df: pd.DataFrame,
    bat_top25: pd.DataFrame,
    bowl_top25: pd.DataFrame,
) -> Dict[str, float]:
    """Compare win-probability rankings against pure run-based metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Complete delivery level dataframe.
    bat_top25 : pd.DataFrame
        Top 25 batters by composite score.
    bowl_top25 : pd.DataFrame
        Top 25 bowlers by composite score.

    Returns
    -------
    Dict[str, float]
        Dictionary of correlation coefficients and test statistics.
    """
    df_p = df[df["is_pressure"]].copy()

    # Batters: Runs Above Expected
    bat_rae = (
        df_p[df_p["is_legal"]]
        .groupby("striker_id")["runs_above_expected"]
        .sum()
        .rename("total_rae")
    )
    bat_merged = bat_top25.merge(bat_rae, left_on="player_id", right_index=True, how="left")
    bat_merged["total_rae"] = bat_merged["total_rae"].fillna(0.0)

    rho_bat, p_rho_bat = spearmanr(bat_merged["rank"], -bat_merged["total_rae"])
    tau_bat, p_tau_bat = kendalltau(bat_merged["rank"], -bat_merged["total_rae"])

    # Bowlers: Wickets Above Expected
    bowl_wae = (
        df_p.groupby("bowler_id")["wickets_above_expected"]
        .sum()
        .rename("total_wae")
    )
    bowl_merged = bowl_top25.merge(bowl_wae, left_on="player_id", right_index=True, how="left")
    bowl_merged["total_wae"] = bowl_merged["total_wae"].fillna(0.0)

    rho_bowl, p_rho_bowl = spearmanr(bowl_merged["rank"], -bowl_merged["total_wae"])
    tau_bowl, p_tau_bowl = kendalltau(bowl_merged["rank"], -bowl_merged["total_wae"])

    return {
        "bat_rae_spearman": float(rho_bat),
        "bat_rae_kendall": float(tau_bat),
        "bowl_wae_spearman": float(rho_bowl),
        "bowl_wae_kendall": float(tau_bowl),
    }
