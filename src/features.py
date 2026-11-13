"""Feature engineering module for the SupeRR pressure analysis pipeline.

This module computes all match-state features strictly BEFORE each delivery.
It incorporates cumulative match state, rolling scoring rates, sequence trackers,
venue par scores and chronological team Elo ratings without data leakage.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_chronological_team_elo(
    matches: pd.DataFrame,
    k_factor: float = 32.0,
    initial_elo: float = 1500.0,
) -> Dict[int, Dict[int, float]]:
    """Compute pre-match Elo ratings for every team chronologically.

    Parameters
    ----------
    matches : pd.DataFrame
        Matches dataframe containing match_date, team ids and winners.
    k_factor : float
        Elo update speed parameter.
    initial_elo : float
        Starting Elo rating for all teams.

    Returns
    -------
    Dict[int, Dict[int, float]]
        Mapping from match_id to dictionary of pre-match team Elo ratings.
    """
    sorted_matches = matches[matches["win_team_id"].notna()].copy()
    sorted_matches["match_date"] = pd.to_datetime(sorted_matches["match_date"], errors="coerce")
    sorted_matches = sorted_matches.sort_values("match_date").reset_index(drop=True)

    current_elo: Dict[int, float] = {}
    pre_match_elo: Dict[int, Dict[int, float]] = {}

    for _, row in sorted_matches.iterrows():
        mid = int(row["match_id"])
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        wt = int(row["win_team_id"])

        r_h = current_elo.get(ht, initial_elo)
        r_a = current_elo.get(at, initial_elo)

        pre_match_elo[mid] = {ht: r_h, at: r_a}

        # Expected score for home team
        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
        actual_h = 1.0 if wt == ht else 0.0

        current_elo[ht] = r_h + k_factor * (actual_h - exp_h)
        current_elo[at] = r_a + k_factor * ((1.0 - actual_h) - (1.0 - exp_h))

    return pre_match_elo


def compute_leave_one_out_venue_par(
    df: pd.DataFrame,
    shrinkage_k: float = 3.0,
    default_par: float = 165.0,
) -> pd.DataFrame:
    """Compute leave-one-out average first innings score per venue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery level dataframe.
    shrinkage_k : float
        Shrinkage weight toward global mean for small sample venues.
    default_par : float
        Fallback first innings par score.

    Returns
    -------
    pd.DataFrame
        Dataframe mapping match_id to leave-one-out venue par score.
    """
    inn1_totals = (
        df[df["innings"] == 1]
        .groupby(["match_id", "venue_id"])["total_ball_runs"]
        .sum()
        .reset_index(name="inn1_total")
    )

    global_mean = inn1_totals["inn1_total"].mean()
    if np.isnan(global_mean):
        global_mean = default_par

    venue_sums = inn1_totals.groupby("venue_id")["inn1_total"].transform("sum")
    venue_counts = inn1_totals.groupby("venue_id")["inn1_total"].transform("count")

    loo_sum = venue_sums - inn1_totals["inn1_total"]
    loo_count = venue_counts - 1

    loo_mean = np.where(loo_count > 0, loo_sum / np.maximum(1, loo_count), global_mean)
    shrunk_par = (loo_mean * loo_count + global_mean * shrinkage_k) / (loo_count + shrinkage_k)

    inn1_totals["venue_par_score"] = np.where(loo_count > 0, shrunk_par, global_mean)
    return inn1_totals[["match_id", "venue_par_score"]]


def build_match_state_features(
    df: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all state and context features strictly before each delivery.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery level dataframe.
    matches : pd.DataFrame
        Matches dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe augmented with pre-ball state features.
    """
    df = df.copy()

    # 1. Chronological Team Elo difference
    elo_lookup = compute_chronological_team_elo(matches)
    def get_elo_diff(row):
        mid = int(row["match_id"])
        bt = int(row["batting_team_id"])
        bwt = int(row["bowling_team_id"])
        match_ratings = elo_lookup.get(mid, {})
        b_elo = match_ratings.get(bt, 1500.0)
        bw_elo = match_ratings.get(bwt, 1500.0)
        return b_elo - bw_elo

    df["elo_diff"] = df.apply(get_elo_diff, axis=1)

    # 2. Leave-one-out Venue Par Score
    venue_pars = compute_leave_one_out_venue_par(df)
    df = df.merge(venue_pars, on="match_id", how="left")
    df["venue_par_score"] = df["venue_par_score"].fillna(165.0)

    # 3. Core Pre-Ball Counters (shifted by 1 within match and innings)
    df["prev_legal"] = df.groupby(["match_id", "innings"])["is_legal"].shift(1).fillna(0).astype(int)
    df["legal_balls_bowled"] = df.groupby(["match_id", "innings"])["prev_legal"].cumsum()

    df["prev_runs"] = df.groupby(["match_id", "innings"])["total_ball_runs"].shift(1).fillna(0).astype(int)
    df["team_runs"] = df.groupby(["match_id", "innings"])["prev_runs"].cumsum()

    df["prev_wkt"] = df.groupby(["match_id", "innings"])["is_wicket"].shift(1).fillna(0).astype(int)
    df["wickets_lost"] = df.groupby(["match_id", "innings"])["prev_wkt"].cumsum()
    df["wickets_in_hand"] = (10 - df["wickets_lost"]).clip(lower=0, upper=10)

    # Allocated balls handling DLS revised overs (which are stored in balls)
    inn1_alloc = np.where(df["first_inn_revised_overs"].notna(), df["first_inn_revised_overs"], 120.0)
    inn2_alloc = np.where(df["second_inn_revised_overs"].notna(), df["second_inn_revised_overs"], 120.0)
    df["total_balls_alloc"] = np.where(df["innings"] == 2, inn2_alloc, inn1_alloc)
    df["balls_remaining"] = np.maximum(0.0, df["total_balls_alloc"] - df["legal_balls_bowled"])

    # Over number and phase
    df["over_number"] = (df["legal_balls_bowled"] // 6 + 1).clip(lower=1, upper=20)
    df["ball_in_over"] = (df["legal_balls_bowled"] % 6 + 1).astype(int)

    # Phase categorization
    phase_conditions = [
        df["legal_balls_bowled"] < 36,
        (df["legal_balls_bowled"] >= 36) & (df["legal_balls_bowled"] < 90),
        df["legal_balls_bowled"] >= 90,
    ]
    phase_choices = ["powerplay", "middle", "death"]
    df["phase"] = np.select(phase_conditions, phase_choices, default="middle")

    # Current run rate (computed after 6 balls, par fallback before)
    overs_bowled = df["legal_balls_bowled"] / 6.0
    df["current_run_rate"] = np.where(df["legal_balls_bowled"] >= 6, df["team_runs"] / overs_bowled, 8.0)

    # Recent scoring rate (rolling sum over previous 12 legal balls)
    df["recent_run_rate"] = df.groupby(["match_id", "innings"])["prev_runs"].transform(
        lambda s: s.rolling(12, min_periods=1).sum() / 2.0
    )

    # Sequence trackers: boundaries and wickets
    df["prev_boundary"] = (df["is_four"] | df["is_six"]).shift(1).fillna(0).astype(int)
    df["boundary_seq"] = df.groupby(["match_id", "innings"])["prev_boundary"].cumsum()
    df["balls_since_last_boundary"] = df["legal_balls_bowled"] - df.groupby(
        ["match_id", "innings", "boundary_seq"]
    )["legal_balls_bowled"].transform("first")

    df["partnership_seq"] = df["wickets_lost"]
    df["partnership_runs"] = df["team_runs"] - df.groupby(
        ["match_id", "innings", "partnership_seq"]
    )["team_runs"].transform("first")
    df["partnership_balls"] = df["legal_balls_bowled"] - df.groupby(
        ["match_id", "innings", "partnership_seq"]
    )["legal_balls_bowled"].transform("first")
    df["balls_since_last_wicket"] = df["partnership_balls"]

    # 4. First innings total for target calculation
    inn1_final_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_ball_runs"]
        .sum()
        .rename("inn1_total")
    )
    df = df.merge(inn1_final_totals, on="match_id", how="left")

    # 5. Innings 2 Chase State
    df["target"] = np.where(
        df["revised_target"].notna() & (df["revised_target"] > 0),
        df["revised_target"],
        df["inn1_total"].fillna(165.0) + 1.0,
    )
    df["runs_required"] = np.maximum(0.0, df["target"] - df["team_runs"])

    overs_remaining = df["balls_remaining"] / 6.0
    df["required_run_rate"] = np.where(
        df["balls_remaining"] > 0,
        df["runs_required"] / np.maximum(0.1, overs_remaining),
        np.where(df["runs_required"] > 0, 99.0, 0.0),
    )
    df["rrr_minus_crr"] = df["required_run_rate"] - df["current_run_rate"]
    df["runs_required_per_wkt"] = df["runs_required"] / np.maximum(1, df["wickets_in_hand"])
    df["required_runs_per_ball"] = df["runs_required"] / np.maximum(1.0, df["balls_remaining"])
    df["resource_proxy"] = (df["balls_remaining"] / 120.0) * ((df["wickets_in_hand"] / 10.0) ** 0.5)

    # 6. Innings 1 State
    wicket_discount = (df["wickets_in_hand"] / 10.0) ** 0.3
    df["projected_score"] = df["team_runs"] + overs_remaining * df["current_run_rate"] * wicket_discount
    df["run_rate_shortfall"] = (
        (df["venue_par_score"] - df["team_runs"]) / np.maximum(1.0, overs_remaining) - df["current_run_rate"]
    )

    # 7. Context variables
    df["batting_order"] = df["batting_order"].fillna(7).astype(int).clip(1, 11)
    df["is_knockout"] = df["knock_out"].fillna(0).astype(int)
    df["is_rhb"] = (df["batting_type"].astype(str).str.upper() == "RHB").astype(int)
    df["is_pace"] = (df["pace_or_spin"].astype(str).str.lower() == "pace").astype(int)

    # Simplify bowling type into broad groups
    bowling_map = {
        "Right-arm Fast": "pace_rf",
        "Right-arm fast-medium": "pace_rfm",
        "Right-arm medium-fast": "pace_rmf",
        "Right-arm medium": "pace_rm",
        "Left-arm Fast": "pace_lf",
        "Left-arm fast-medium": "pace_lfm",
        "Left-arm medium-fast": "pace_lmf",
        "Left-arm medium": "pace_lm",
        "Right-arm offbreak": "spin_off",
        "Right-arm Leg-Break": "spin_leg",
        "Slow left-arm orthodox": "spin_sla",
        "Left-arm Chinaman": "spin_chinaman",
    }
    df["bowling_group"] = df["bowling_type"].map(bowling_map).fillna("other")

    # Win target label for win probability models
    df["batting_side_win"] = (df["batting_team_id"] == df["win_team_id"]).astype(int)

    return df
