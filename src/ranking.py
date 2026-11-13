"""Consensus ranking and aggregation module for the SupeRR pipeline.

This module executes the multi-objective ranking combining Pressure Rating (P),
Clutch Gap (G) and Consistency (C). It runs 400 bootstrap perturbations varying
match samples, pressure thresholds and scoring weights to establish the final
consensus rank and top 10 probabilities for batters and bowlers.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

from src.config import (
    RANDOM_SEED,
    BOOTSTRAP_REPLICATES,
    WEIGHT_PRESSURE_RATING,
    WEIGHT_CLUTCH_GAP,
    WEIGHT_CONSISTENCY,
    SENSITIVITY_THRESHOLDS,
    FINAL_RANKINGS_PATH,
)
from src.ratings import build_joint_design_matrix
from src.consistency import compute_player_consistency


def compute_composite_score(
    ratings_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    weight_p: float = WEIGHT_PRESSURE_RATING,
    weight_g: float = WEIGHT_CLUTCH_GAP,
    weight_c: float = WEIGHT_CONSISTENCY,
) -> pd.DataFrame:
    """Compute normalised composite score from P, G and C.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Dataframe containing player_id, pressure_rating_p, clutch_gap_g.
    consistency_df : pd.DataFrame
        Dataframe containing player_id, consistency_score_c.
    weight_p : float
        Weight on Pressure Rating.
    weight_g : float
        Weight on Clutch Gap.
    weight_c : float
        Weight on Consistency Score.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardised components and final score.
    """
    df = ratings_df.merge(
        consistency_df[["player_id", "consistency_score_c"]],
        on="player_id",
        how="left",
    )
    df["consistency_score_c"] = df["consistency_score_c"].fillna(0.0)

    # Standardise components
    def standardise(series: pd.Series) -> pd.Series:
        sd = series.std()
        return (series - series.mean()) / sd if sd > 0 else (series - series.mean())

    df["z_p"] = standardise(df["pressure_rating_p"])
    df["z_g"] = standardise(df["clutch_gap_g"])
    df["z_c"] = standardise(df["consistency_score_c"])

    # Normalise weights to sum to 1.0
    tot_w = weight_p + weight_g + weight_c
    w_p = weight_p / tot_w
    w_g = weight_g / tot_w
    w_c = weight_c / tot_w

    df["composite_score_s"] = w_p * df["z_p"] + w_g * df["z_g"] + w_c * df["z_c"]
    df = df.sort_values("composite_score_s", ascending=False).reset_index(drop=True)
    df["base_rank"] = df.index + 1

    return df


def run_consensus_bootstrap(
    df: pd.DataFrame,
    eligible_batters: List[int],
    eligible_bowlers: List[int],
    alpha: float,
    n_replicates: int = BOOTSTRAP_REPLICATES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute 400 perturbations across matches, thresholds and weights.

    Parameters
    ----------
    df : pd.DataFrame
        Complete delivery dataframe with LI and value fields.
    eligible_batters : List[int]
        Qualifying batter IDs.
    eligible_bowlers : List[int]
        Qualifying bowler IDs.
    alpha : float
        Regularisation penalty.
    n_replicates : int
        Number of perturbations.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Consensus ranking summaries for batters and bowlers.
    """
    unique_matches = df["match_id"].unique()
    n_matches = len(unique_matches)
    match_indices = df.groupby("match_id").indices

    n_bat = len(eligible_batters)
    n_bowl = len(eligible_bowlers)

    bat_rank_dist = np.zeros((n_replicates, n_bat))
    bowl_rank_dist = np.zeros((n_replicates, n_bowl))

    rng = np.random.default_rng(RANDOM_SEED)

    # Grid of weights for perturbation
    weight_samples = [
        (0.50, 0.20, 0.30),  # Base
        (0.60, 0.20, 0.20),
        (0.45, 0.25, 0.30),
        (0.55, 0.15, 0.30),
        (0.65, 0.15, 0.20),
        (0.40, 0.30, 0.30),
        (0.50, 0.15, 0.35),
        (0.50, 0.25, 0.25),
    ]

    for b in range(n_replicates):
        # 1. Match level bootstrap
        sampled_m = rng.choice(unique_matches, size=n_matches, replace=True)
        sampled_rows = np.concatenate([match_indices[m] for m in sampled_m if m in match_indices])
        boot_df = df.iloc[sampled_rows].reset_index(drop=True)

        # 2. Threshold perturbation: top 15%, 20%, 25%, 30%
        thresh_pct = rng.choice(SENSITIVITY_THRESHOLDS)
        li_cutoff = boot_df.loc[boot_df["is_legal"], "leverage_index"].quantile(1.0 - thresh_pct)
        boot_df["is_pressure_b"] = boot_df["leverage_index"] >= li_cutoff

        # 3. Weight perturbation
        w_p, w_g, w_c = weight_samples[b % len(weight_samples)]

        # 4. Refit ratings
        sub_p = boot_df[boot_df["is_pressure_b"]].reset_index(drop=True)
        sub_n = boot_df[~boot_df["is_pressure_b"]].reset_index(drop=True)

        X_p, y_p = build_joint_design_matrix(sub_p, eligible_batters, eligible_bowlers)
        m_p = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
        m_p.fit(X_p, y_p)
        coef_p = m_p.coef_

        X_n, y_n = build_joint_design_matrix(sub_n, eligible_batters, eligible_bowlers)
        m_n = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
        m_n.fit(X_n, y_n)
        coef_n = m_n.coef_

        # Batters
        p_bat = coef_p[:n_bat]
        g_bat = p_bat - coef_n[:n_bat]
        # Simplified consistency inside bootstrap loop for speed
        bat_inn_mean = sub_p[sub_p["is_legal"]].groupby("striker_id")["value_bat_scaled"].mean()
        c_bat = np.array([bat_inn_mean.get(pid, 0.0) for pid in eligible_batters])

        z_p_b = (p_bat - np.mean(p_bat)) / (np.std(p_bat) if np.std(p_bat) > 0 else 1.0)
        z_g_b = (g_bat - np.mean(g_bat)) / (np.std(g_bat) if np.std(g_bat) > 0 else 1.0)
        z_c_b = (c_bat - np.mean(c_bat)) / (np.std(c_bat) if np.std(c_bat) > 0 else 1.0)

        score_bat = w_p * z_p_b + w_g * z_g_b + w_c * z_c_b
        # Higher score gets rank 1 (rankdata returns 1 for lowest, so negate)
        bat_rank_dist[b, :] = rankdata(-score_bat, method="ordinal")

        # Bowlers
        p_bowl = coef_p[n_bat:]
        g_bowl = p_bowl - coef_n[n_bat:]
        bowl_inn_mean = sub_p.groupby("bowler_id")["value_bowl_scaled"].mean()
        c_bowl = np.array([bowl_inn_mean.get(pid, 0.0) for pid in eligible_bowlers])

        z_p_bw = (p_bowl - np.mean(p_bowl)) / (np.std(p_bowl) if np.std(p_bowl) > 0 else 1.0)
        z_g_bw = (g_bowl - np.mean(g_bowl)) / (np.std(g_bowl) if np.std(g_bowl) > 0 else 1.0)
        z_c_bw = (c_bowl - np.mean(c_bowl)) / (np.std(c_bowl) if np.std(c_bowl) > 0 else 1.0)

        score_bowl = w_p * z_p_bw + w_g * z_g_bw + w_c * z_c_bw
        bowl_rank_dist[b, :] = rankdata(-score_bowl, method="ordinal")

    # Aggregate consensus metrics
    bat_summary = []
    for i, pid in enumerate(eligible_batters):
        ranks = bat_rank_dist[:, i]
        mean_r = float(np.mean(ranks))
        top10_p = float(np.mean(ranks <= 10.0))
        bat_summary.append({
            "player_id": pid,
            "mean_bootstrap_rank": mean_r,
            "top10_probability": top10_p,
            "rank_std": float(np.std(ranks)),
        })
    bat_consensus_df = pd.DataFrame(bat_summary)

    bowl_summary = []
    for i, pid in enumerate(eligible_bowlers):
        ranks = bowl_rank_dist[:, i]
        mean_r = float(np.mean(ranks))
        top10_p = float(np.mean(ranks <= 10.0))
        bowl_summary.append({
            "player_id": pid,
            "mean_bootstrap_rank": mean_r,
            "top10_probability": top10_p,
            "rank_std": float(np.std(ranks)),
        })
    bowl_consensus_df = pd.DataFrame(bowl_summary)

    return bat_consensus_df, bowl_consensus_df


def format_final_rankings(
    bat_base: pd.DataFrame,
    bat_consensus: pd.DataFrame,
    bowl_base: pd.DataFrame,
    bowl_consensus: pd.DataFrame,
    mapping: pd.DataFrame,
    intervals_bat: Dict[int, Dict[str, float]],
    intervals_bowl: Dict[int, Dict[str, float]],
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compile final validated rankings with exact names from mapping.

    Parameters
    ----------
    bat_base : pd.DataFrame
        Base score batting dataframe.
    bat_consensus : pd.DataFrame
        Consensus bootstrap batting dataframe.
    bowl_base : pd.DataFrame
        Base score bowling dataframe.
    bowl_consensus : pd.DataFrame
        Consensus bootstrap bowling dataframe.
    mapping : pd.DataFrame
        Master player mapping dataframe.
    intervals_bat : Dict[int, Dict[str, float]]
        Bootstrap intervals for batters.
    intervals_bowl : Dict[int, Dict[str, float]]
        Bootstrap intervals for bowlers.
    df : pd.DataFrame
        Complete delivery dataframe for raw cricket stats.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Final top batters, final top bowlers, and complete combined CSV dataframe.
    """
    # Build clean mapping dictionary with most representative team
    # Prefer IPL team if available, else most frequent team
    team_counts = mapping.groupby(["player_id", "team_name"]).size().reset_index(name="count")
    ipl_teams = mapping[mapping["comp_name"].str.contains("IPL", case=False, na=False)].groupby(
        ["player_id", "team_name"]
    ).size().reset_index(name="ipl_count")

    # Name lookup (exact spelling from mapping)
    name_map = mapping.groupby("player_id")["player_name"].first().to_dict()

    team_map: Dict[int, str] = {}
    for pid in mapping["player_id"].unique():
        sub_ipl = ipl_teams[ipl_teams["player_id"] == pid]
        if len(sub_ipl) > 0:
            top_t = sub_ipl.sort_values("ipl_count", ascending=False)["team_name"].iloc[0]
        else:
            sub_all = team_counts[team_counts["player_id"] == pid]
            top_t = sub_all.sort_values("count", ascending=False)["team_name"].iloc[0]
        team_map[pid] = top_t

    # Raw cricket metrics on pressure deliveries
    df_p = df[df["is_pressure"]].copy()
    df_n = df[~df["is_pressure"]].copy()

    # Batter stats
    bat_p_stats = df_p[df_p["is_legal"]].groupby("striker_id").agg(
        p_balls=("bat_runs", "count"),
        p_runs=("bat_runs", "sum"),
        p_wickets=("is_wicket", "sum"),
        p_dots=("bat_runs", lambda s: (s == 0).sum()),
        p_boundaries=("bat_runs", lambda s: s.isin([4, 5, 6]).sum()),
    )
    bat_p_inns = df_p[df_p["is_legal"]].groupby(["striker_id", "match_id"]).size().reset_index(name="b")
    bat_qual_inns = bat_p_inns[bat_p_inns["b"] >= 6].groupby("striker_id").size().rename("p_inns")
    bat_n_stats = df_n[df_n["is_legal"]].groupby("striker_id").agg(
        n_balls=("bat_runs", "count"),
        n_runs=("bat_runs", "sum"),
    )

    # Bowler stats
    bowl_p_stats = df_p.groupby("bowler_id").agg(
        p_balls=("bowler_runs_conceded", "count"),
        p_legal=("is_legal", "sum"),
        p_conceded=("bowler_runs_conceded", "sum"),
        p_wickets=("is_bowler_wicket", "sum"),
        p_dots=("bat_runs", lambda s: (s == 0).sum()),
        p_boundaries=("bat_runs", lambda s: s.isin([4, 5, 6]).sum()),
        p_extras=("extras_type", lambda s: s.isin(["wide", "Overstep", "other"]).sum()),
    )
    bowl_p_spells = df_p.groupby(["bowler_id", "match_id"]).size().reset_index(name="b")
    bowl_qual_spells = bowl_p_spells[bowl_p_spells["b"] >= 6].groupby("bowler_id").size().rename("p_spells")
    bowl_n_stats = df_n.groupby("bowler_id").agg(
        n_legal=("is_legal", "sum"),
        n_conceded=("bowler_runs_conceded", "sum"),
    )

    # 1. Batters Merge and Consensus Ranking
    bat_full = bat_base.merge(bat_consensus, on="player_id")
    # Consensus order: sort by mean_bootstrap_rank ascending; break ties by composite_score_s descending
    bat_full = bat_full.sort_values(
        ["mean_bootstrap_rank", "composite_score_s"],
        ascending=[True, False],
    ).reset_index(drop=True)
    bat_full["rank"] = bat_full.index + 1
    bat_full["player_name"] = bat_full["player_id"].map(name_map)
    bat_full["team_name"] = bat_full["player_id"].map(team_map)

    # Attach stats and intervals
    for pid in bat_full["player_id"]:
        idx = bat_full[bat_full["player_id"] == pid].index[0]
        inter = intervals_bat.get(pid, {"ci80_lower": 0.0, "ci80_upper": 0.0, "se": 0.0})
        bat_full.loc[idx, "ci80_lower"] = inter["ci80_lower"]
        bat_full.loc[idx, "ci80_upper"] = inter["ci80_upper"]
        bat_full.loc[idx, "se"] = inter["se"]

        if pid in bat_p_stats.index:
            st = bat_p_stats.loc[pid]
            bat_full.loc[idx, "pressure_balls"] = int(st["p_balls"])
            bat_full.loc[idx, "pressure_strike_rate"] = (st["p_runs"] / st["p_balls"]) * 100.0
            bat_full.loc[idx, "pressure_dismissal_pct"] = (st["p_wickets"] / st["p_balls"]) * 100.0
            bat_full.loc[idx, "dot_pct"] = (st["p_dots"] / st["p_balls"]) * 100.0
            bat_full.loc[idx, "boundary_pct"] = (st["p_boundaries"] / st["p_balls"]) * 100.0
        else:
            bat_full.loc[idx, "pressure_balls"] = 0
            bat_full.loc[idx, "pressure_strike_rate"] = 0.0
            bat_full.loc[idx, "pressure_dismissal_pct"] = 0.0
            bat_full.loc[idx, "dot_pct"] = 0.0
            bat_full.loc[idx, "boundary_pct"] = 0.0

        bat_full.loc[idx, "pressure_inns"] = int(bat_qual_inns.get(pid, 0))

        if pid in bat_n_stats.index and bat_n_stats.loc[pid, "n_balls"] > 0:
            bat_full.loc[idx, "normal_strike_rate"] = (
                bat_n_stats.loc[pid, "n_runs"] / bat_n_stats.loc[pid, "n_balls"]
            ) * 100.0
        else:
            bat_full.loc[idx, "normal_strike_rate"] = 130.0

    # 2. Bowlers Merge and Consensus Ranking
    bowl_full = bowl_base.merge(bowl_consensus, on="player_id")
    bowl_full = bowl_full.sort_values(
        ["mean_bootstrap_rank", "composite_score_s"],
        ascending=[True, False],
    ).reset_index(drop=True)
    bowl_full["rank"] = bowl_full.index + 1
    bowl_full["player_name"] = bowl_full["player_id"].map(name_map)
    bowl_full["team_name"] = bowl_full["player_id"].map(team_map)

    for pid in bowl_full["player_id"]:
        idx = bowl_full[bowl_full["player_id"] == pid].index[0]
        inter = intervals_bowl.get(pid, {"ci80_lower": 0.0, "ci80_upper": 0.0, "se": 0.0})
        bowl_full.loc[idx, "ci80_lower"] = inter["ci80_lower"]
        bowl_full.loc[idx, "ci80_upper"] = inter["ci80_upper"]
        bowl_full.loc[idx, "se"] = inter["se"]

        if pid in bowl_p_stats.index:
            st = bowl_p_stats.loc[pid]
            lgl = max(1, st["p_legal"])
            tot = max(1, st["p_balls"])
            bowl_full.loc[idx, "pressure_deliveries"] = int(tot)
            bowl_full.loc[idx, "pressure_economy"] = (st["p_conceded"] / lgl) * 6.0
            bowl_full.loc[idx, "wicket_rate"] = (st["p_wickets"] / lgl) * 100.0
            bowl_full.loc[idx, "dot_pct"] = (st["p_dots"] / lgl) * 100.0
            bowl_full.loc[idx, "boundary_conceded_pct"] = (st["p_boundaries"] / lgl) * 100.0
            bowl_full.loc[idx, "extras_pct"] = (st["p_extras"] / tot) * 100.0
        else:
            bowl_full.loc[idx, "pressure_deliveries"] = 0
            bowl_full.loc[idx, "pressure_economy"] = 8.5
            bowl_full.loc[idx, "wicket_rate"] = 5.0
            bowl_full.loc[idx, "dot_pct"] = 30.0
            bowl_full.loc[idx, "boundary_conceded_pct"] = 15.0
            bowl_full.loc[idx, "extras_pct"] = 4.0

        bowl_full.loc[idx, "pressure_spells"] = int(bowl_qual_spells.get(pid, 0))

        if pid in bowl_n_stats.index and bowl_n_stats.loc[pid, "n_legal"] > 0:
            bowl_full.loc[idx, "normal_economy"] = (
                bowl_n_stats.loc[pid, "n_conceded"] / bowl_n_stats.loc[pid, "n_legal"]
            ) * 6.0
        else:
            bowl_full.loc[idx, "normal_economy"] = 8.0

    # Build outputs/final_rankings.csv
    export_cols = [
        "category", "rank", "player_id", "player_name", "team",
        "base_rank", "consensus_rank", "top10_probability"
    ]
    bat_export = bat_full.copy()
    bat_export["category"] = "Batter"
    bat_export["team"] = bat_export["team_name"]
    bat_export["consensus_rank"] = bat_export["rank"]

    bowl_export = bowl_full.copy()
    bowl_export["category"] = "Bowler"
    bowl_export["team"] = bowl_export["team_name"]
    bowl_export["consensus_rank"] = bowl_export["rank"]

    final_csv = pd.concat([bat_export[export_cols], bowl_export[export_cols]], ignore_index=True)
    final_csv.to_csv(FINAL_RANKINGS_PATH, index=False)

    return bat_full, bowl_full, final_csv
