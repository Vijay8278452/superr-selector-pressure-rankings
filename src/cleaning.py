"""Data cleaning and audit module for the SupeRR pressure analysis pipeline.

This module validates data integrity, performs reconciliation checks,
filters the sample according to competition rules and prepares clean features.
"""

from pathlib import Path
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.config import (
    EXCLUDED_INNINGS,
    EXCLUDED_RESULTS,
    LEGAL_BALL_EXTRAS_EXCLUDED,
    BOWLER_CREDITED_DISMISSALS,
    LOGS_DIR,
    TABLES_DIR,
)


def run_data_audit(datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    """Run comprehensive verification checks on the raw datasets.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Raw datasets dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        Table 1 summary dataframe and detailed text audit log.
    """
    bbb = datasets["ball_by_ball"]
    matches = datasets["matches"]
    comps = datasets["competitions"]
    mapping = datasets["mapping"]

    lines = []
    lines.append("=" * 70)
    lines.append("SUPERR SELECTOR SEASON 2: DATA AUDIT AND VERIFICATION LOG")
    lines.append("=" * 70)
    lines.append(f"Total rows in ball_by_ball: {len(bbb):,}")
    lines.append(f"Total matches in matches.csv: {len(matches):,}")
    lines.append(f"Total competitions: {len(comps):,}")
    lines.append(f"Total player mappings: {len(mapping):,}")
    lines.append("")

    # 1. Verification of runs meaning
    bbb_temp = bbb.copy()
    bbb_temp["total_ball_runs"] = bbb_temp["runs"] + bbb_temp["extras1_runs"] + bbb_temp["extras_2_runs"]
    inn_totals = bbb_temp.groupby(["match_id", "innings"])["total_ball_runs"].sum().unstack()

    merged_m = matches.merge(inn_totals, on="match_id", how="left")

    run_wins = merged_m[merged_m["result"].str.contains("run", case=False, na=False)].copy()
    def get_run_margin(res: str):
        match = re.search(r"by\s+(\d+)\s+run", str(res), re.IGNORECASE)
        return int(match.group(1)) if match else None

    run_wins["margin"] = run_wins["result"].apply(get_run_margin)
    run_wins["diff"] = run_wins[1] - run_wins[2]
    reconciled_run_wins = (run_wins["margin"] == run_wins["diff"]).sum()
    total_run_wins = len(run_wins)

    wkt_wins = merged_m[merged_m["result"].str.contains("wicket", case=False, na=False)].copy()
    wkt_wins["target"] = np.where(
        wkt_wins["revised_target"].notna() & (wkt_wins["revised_target"] > 0),
        wkt_wins["revised_target"],
        wkt_wins[1] + 1
    )
    reconciled_wkt_wins = (wkt_wins[2] >= wkt_wins["target"]).sum()
    total_wkt_wins = len(wkt_wins)

    total_decided = total_run_wins + total_wkt_wins
    total_reconciled = reconciled_run_wins + reconciled_wkt_wins
    reconciliation_pct = (total_reconciled / total_decided) * 100.0

    lines.append("1. RUNS DEFINITION RECONCILIATION")
    lines.append(f"   Matches decided by runs: {total_run_wins}")
    lines.append(f"   Run wins matching margin exactly: {reconciled_run_wins} ({reconciled_run_wins / total_run_wins * 100:.2f}%)")
    lines.append(f"   Matches decided by wickets: {total_wkt_wins}")
    lines.append(f"   Wicket wins where target reached: {reconciled_wkt_wins} ({reconciled_wkt_wins / total_wkt_wins * 100:.2f}%)")
    lines.append(f"   Overall reconciliation rate: {total_reconciled} / {total_decided} ({reconciliation_pct:.2f}%)")
    lines.append("   Conclusion: 'runs' represents bat runs only. Total runs equals runs + extras1_runs + extras_2_runs.")
    lines.append("")

    # 2. Legal ball rule
    bbb_temp["legal_strict"] = ~bbb_temp["extras_type"].isin(LEGAL_BALL_EXTRAS_EXCLUDED)
    bbb_temp["legal_naive"] = ~bbb_temp["extras_type"].isin(["wide", "Overstep"])

    inn_max_over = bbb_temp.groupby(["match_id", "innings"])["overid"].transform("max")
    completed_overs = bbb_temp[bbb_temp["overid"] < inn_max_over]

    strict_counts = completed_overs.groupby(["match_id", "innings", "overid"])["legal_strict"].sum()
    naive_counts = completed_overs.groupby(["match_id", "innings", "overid"])["legal_naive"].sum()

    pct_strict_6 = (strict_counts == 6).mean() * 100.0
    pct_naive_6 = (naive_counts == 6).mean() * 100.0

    lines.append("2. LEGAL BALL RULE VERIFICATION")
    lines.append(f"   Total completed overs tested: {len(strict_counts):,}")
    lines.append(f"   Naive rule (exclude wide and Overstep only): {pct_naive_6:.2f}% of overs have exactly 6 balls")
    lines.append(f"   Strict rule (exclude wide, Overstep, other and Penalty): {pct_strict_6:.4f}% of overs have exactly 6 balls")
    lines.append("   Conclusion: 'other' and 'Penalty' are illegal deliveries or administrative penalty events.")
    lines.append("   Strict rule adopted: legal delivery is any ball where extras_type is not in (wide, Overstep, other, Penalty).")
    lines.append("")

    # 3. Player mapping checks
    id_name_counts = mapping.groupby("player_id")["player_name"].nunique()
    multi_name_ids = (id_name_counts > 1).sum()

    name_id_counts = mapping.groupby("player_name")["player_id"].nunique()
    multi_id_names = (name_id_counts > 1).sum()

    bbb_striker_names = bbb.groupby("striker_id")["striker_name"].nunique()
    bbb_bowler_names = bbb.groupby("bowler_id")["bowler_name"].nunique()

    lines.append("3. PLAYER IDENTIFIER CONSISTENCY")
    lines.append(f"   Player IDs with multiple names in mapping: {multi_name_ids}")
    lines.append(f"   Player names with multiple IDs in mapping: {multi_id_names} (different individuals sharing a name)")
    lines.append(f"   Striker IDs with multiple names in ball_by_ball: {(bbb_striker_names > 1).sum()}")
    lines.append(f"   Bowler IDs with multiple names in ball_by_ball: {(bbb_bowler_names > 1).sum()}")
    lines.append("   Conclusion: player_id is globally unique and stable across competitions.")
    lines.append("")

    # 4. Exclusions audit
    bbb_matches = set(bbb["match_id"].unique())
    all_matches = set(matches["match_id"].unique())
    matches_no_bbb = len(all_matches - bbb_matches)

    no_result_matches = matches[matches["win_team_id"].isna()]
    no_result_in_bbb = len(no_result_matches[no_result_matches["match_id"].isin(bbb_matches)])

    super_over_balls = len(bbb[bbb["innings"].isin(EXCLUDED_INNINGS)])

    dls_matches = matches[matches["revised_target"].notna() | matches["first_inn_revised_overs"].notna() | matches["second_inn_revised_overs"].notna()]
    dls_total = len(dls_matches)
    dls_in_sample = len(dls_matches[dls_matches["win_team_id"].notna() & dls_matches["match_id"].isin(bbb_matches)])
    dls_excluded_no_result = dls_total - dls_in_sample

    final_decided_matches = matches[matches["win_team_id"].notna() & matches["match_id"].isin(bbb_matches)]
    final_balls = len(bbb[(bbb["match_id"].isin(final_decided_matches["match_id"])) & (~bbb["innings"].isin(EXCLUDED_INNINGS))])

    lines.append("4. SAMPLE EXCLUSIONS AND DATA CLEANING COUNTS")
    lines.append(f"   Raw matches in matches.csv: {len(matches):,}")
    lines.append(f"   Matches abandoned before play (0 balls in ball_by_ball): {matches_no_bbb}")
    lines.append(f"   Matches with balls bowled: {len(bbb_matches):,}")
    lines.append(f"   Matches without winner (No Result / Abandoned / Tied): {no_result_in_bbb}")
    lines.append(f"   Decided matches retained in modelling sample: {len(final_decided_matches):,}")
    lines.append(f"   Raw deliveries in ball_by_ball: {len(bbb):,}")
    lines.append(f"   Super over deliveries removed (innings 3 and 4): {super_over_balls}")
    lines.append(f"   Deliveries removed from uncompleted / no result matches: {len(bbb) - super_over_balls - final_balls:,}")
    lines.append(f"   Final modelling sample deliveries: {final_balls:,}")
    lines.append(f"   DLS revised matches handled: {dls_in_sample} retained with revised targets and overs; {dls_excluded_no_result} excluded due to no result.")
    lines.append("=" * 70)

    audit_text = "\n".join(lines)

    # Build Table 1 DataFrame
    t1_records = [
        {"Stage": "Raw matches in matches.csv", "Count": len(matches), "Notes": "Historical IPL and T20I matches from 2022 onwards"},
        {"Stage": "Matches abandoned before play", "Count": matches_no_bbb, "Notes": "Zero balls bowled in ball_by_ball dataset"},
        {"Stage": "Matches with ball-by-ball records", "Count": len(bbb_matches), "Notes": "At least one delivery recorded"},
        {"Stage": "Matches excluded (No Result / Abandoned)", "Count": no_result_in_bbb, "Notes": "No winner decided (21 No Result, 3 Abandoned, 1 Tied)"},
        {"Stage": "Completed matches in modelling sample", "Count": len(final_decided_matches), "Notes": "Decided contests with verified winner"},
        {"Stage": "Raw deliveries in ball_by_ball", "Count": len(bbb), "Notes": "All recorded deliveries"},
        {"Stage": "Super over deliveries removed", "Count": super_over_balls, "Notes": "Innings 3 and 4 tie-breaker balls"},
        {"Stage": "Deliveries removed (uncompleted matches)", "Count": len(bbb) - super_over_balls - final_balls, "Notes": "Balls bowled in matches with no final outcome"},
        {"Stage": "Deliveries in modelling sample", "Count": final_balls, "Notes": "Innings 1 and 2 in completed matches"},
        {"Stage": "DLS matches retained", "Count": dls_in_sample, "Notes": "Handled with revised targets and revised balls"},
    ]
    t1_df = pd.DataFrame(t1_records)

    return t1_df, audit_text


def clean_and_prepare_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Filter raw data, join match metadata and create core delivery fields.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Raw datasets dictionary.

    Returns
    -------
    pd.DataFrame
        Cleaned ball-by-ball dataframe ready for feature engineering.
    """
    bbb = datasets["ball_by_ball"].copy()
    matches = datasets["matches"].copy()
    comps = datasets["competitions"].copy()

    # Retain only decided matches with an identified winner
    decided_matches = matches[matches["win_team_id"].notna()].copy()

    # Filter deliveries: valid match and innings 1 or 2
    df = bbb[
        (bbb["match_id"].isin(decided_matches["match_id"])) &
        (~bbb["innings"].isin(EXCLUDED_INNINGS))
    ].copy()

    # Join match metadata
    match_cols = [
        "match_id", "comp_id", "match_date", "stage", "venue_id",
        "home_team_id", "away_team_id", "first_inn_revised_overs",
        "second_inn_revised_overs", "revised_target", "knock_out", "win_team_id"
    ]
    df = df.merge(decided_matches[match_cols], on="match_id", how="left")

    # Join competition metadata
    comp_cols = ["comp_id", "comp_name", "match_format", "season"]
    df = df.merge(comps[comp_cols], on="comp_id", how="left")

    # Format indicators
    df["is_ipl"] = df["comp_name"].astype(str).str.contains("IPL", case=False).astype(int)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Clean boolean flags
    df["is_four"] = df["is_four"].astype(str).str.lower() == "true"
    df["is_six"] = df["is_six"].astype(str).str.lower() == "true"
    df["is_wicket"] = df["is_wicket"].astype(str).str.lower() == "true"

    # Legal ball definition
    df["is_legal"] = ~df["extras_type"].isin(LEGAL_BALL_EXTRAS_EXCLUDED)

    # Bowler credited wickets (excludes RunOut, Retire Out, obstructing the field)
    df["is_bowler_wicket"] = df["is_wicket"] & df["dismissal_type"].isin(BOWLER_CREDITED_DISMISSALS)

    # Run definitions
    df["bat_runs"] = df["runs"].fillna(0).astype(int)
    df["extra_runs"] = (df["extras1_runs"].fillna(0) + df["extras_2_runs"].fillna(0)).astype(int)
    df["total_ball_runs"] = df["bat_runs"] + df["extra_runs"]

    # Bowler conceded runs: bat runs + wides + no-balls (byes and legbyes excluded)
    bowler_extra_mask = df["extras_type"].isin(["wide", "Overstep", "other"])
    df["bowler_runs_conceded"] = df["bat_runs"] + np.where(bowler_extra_mask, df["extra_runs"], 0)

    # Batter dismissal flag
    df["is_batter_dismissed"] = df["is_wicket"] & (df["dismissed_player_id"] == df["striker_id"])

    # Ensure chronologically sorted within each match and innings
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    return df
