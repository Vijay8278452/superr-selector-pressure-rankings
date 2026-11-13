"""Master execution script for the SupeRR pressure analysis pipeline.

This script executes the complete end-to-end analytical workflow:
data auditing, cleaning, feature engineering, win probability estimation,
ball outcome modelling, leverage index calculation, performance attribution,
joint player ratings, consistency scoring, consensus ranking, sensitivity
analysis, table generation and publication figure rendering.
"""

from pathlib import Path
import time
import pandas as pd

from src.config import (
    LOGS_DIR,
    BATTER_MIN_PRESSURE_BALLS,
    BATTER_MIN_PRESSURE_INNINGS,
    BOWLER_MIN_PRESSURE_BALLS,
    BOWLER_MIN_PRESSURE_SPELLS,
)
from src.data_loading import load_raw_datasets
from src.cleaning import run_data_audit, clean_and_prepare_dataset
from src.features import build_match_state_features
from src.win_probability import train_win_probability_models
from src.outcome_model import train_ball_outcome_model
from src.pressure import compute_leverage_index, compute_tier_outcome_table
from src.performance import compute_performance_attribution
from src.ratings import fit_player_ratings, compute_match_cluster_bootstrap
from src.consistency import compute_player_consistency, run_reliability_checks
from src.ranking import compute_composite_score, run_consensus_bootstrap, format_final_rankings
from src.robustness import evaluate_sensitivity_scenarios, compare_with_run_based_metrics
from src.figures import generate_all_figures
from src.tables import export_all_tables


def main():
    """Run the entire analytical pipeline end to end."""
    start_time = time.time()
    print("=" * 70)
    print("SUPERR SELECTOR SEASON 2: REPRODUCIBLE PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    print("\n[Step 1/12] Loading raw datasets...")
    datasets = load_raw_datasets()
    print(f"Loaded datasets in {time.time() - t0:.2f}s")

    # 2. Data Audit and Verification
    t0 = time.time()
    print("\n[Step 2/12] Running data verification audit and reconciliation...")
    t1_audit, audit_log_text = run_data_audit(datasets)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "data_audit.txt", "w", encoding="utf-8") as f:
        f.write(audit_log_text)
    print(f"Data audit completed in {time.time() - t0:.2f}s. Saved to outputs/logs/data_audit.txt")

    # 3. Clean and Prepare
    t0 = time.time()
    print("\n[Step 3/12] Filtering modelling sample and preparing core fields...")
    df = clean_and_prepare_dataset(datasets)
    print(f"Prepared {len(df):,} deliveries across {df['match_id'].nunique():,} matches in {time.time() - t0:.2f}s")

    # 4. Feature Engineering
    t0 = time.time()
    print("\n[Step 4/12] Computing pre-ball match state, Elo and par features...")
    df = build_match_state_features(df, datasets["matches"])
    print(f"Generated match state features in {time.time() - t0:.2f}s")

    # 5. Win Probability Models
    t0 = time.time()
    print("\n[Step 5/12] Fitting out-of-fold calibrated win probability models...")
    df, wp_metrics = train_win_probability_models(df)
    print(f"Innings 2 AUC: {wp_metrics['inn2_hgb_auc']:.4f}, Log Loss: {wp_metrics['inn2_hgb_logloss']:.4f}")
    print(f"Innings 1 AUC: {wp_metrics['inn1_hgb_auc']:.4f}, Log Loss: {wp_metrics['inn1_hgb_logloss']:.4f}")
    print(f"Win probability modelling completed in {time.time() - t0:.2f}s")

    # 6. Multiclass Ball Outcome Model
    t0 = time.time()
    print("\n[Step 6/12] Fitting out-of-fold context-only ball outcome model...")
    df, outcome_metrics = train_ball_outcome_model(df)
    print(f"Outcome Model Log Loss: {outcome_metrics['outcome_model_logloss']:.4f} (Naive baseline: {outcome_metrics['naive_phase_logloss']:.4f})")
    print(f"Ball outcome modelling completed in {time.time() - t0:.2f}s")

    # 7. Leverage Index and Pressure Tiers
    t0 = time.time()
    print("\n[Step 7/12] Computing Leverage Index (LI) and rule-based Pressure Index (PI)...")
    df, pressure_metrics = compute_leverage_index(df, wp_metrics)
    t3_tiers, tier_tests = compute_tier_outcome_table(df)
    print(f"Leverage Index mean: {df.loc[df['is_legal'], 'leverage_index'].mean():.3f}")
    print(f"Pressure Tier cut-offs: Med={pressure_metrics['cut_med']:.2f}, High={pressure_metrics['cut_high']:.2f}")
    print(f"PI to LI Correlation: {pressure_metrics['pi_correlation']:.4f}, R2: {pressure_metrics['pi_r2']:.4f}")
    print(f"Pressure calculations completed in {time.time() - t0:.2f}s")

    # 8. Performance Attribution
    t0 = time.time()
    print("\n[Step 8/12] Computing WPAA and neutral value per ball (V = WPAA / LI)...")
    df = compute_performance_attribution(df)
    print(f"Performance attribution completed in {time.time() - t0:.2f}s")

    # 9. Eligibility Filtering
    t0 = time.time()
    print("\n[Step 9/12] Determining player eligibility pools under pressure...")
    df_p = df[df["is_pressure"]].copy()

    # Batters
    bat_counts = df_p[df_p["is_legal"]].groupby("striker_id").agg(
        pressure_balls=("bat_runs", "count"),
    ).reset_index()
    bat_inns = df_p[df_p["is_legal"]].groupby(["striker_id", "match_id"]).size().reset_index(name="b")
    bat_qual_inns = bat_inns[bat_inns["b"] >= 6].groupby("striker_id").size().reset_index(name="pressure_innings")
    bat_merged = bat_counts.merge(bat_qual_inns, on="striker_id", how="left").fillna(0)

    eligible_batters = bat_merged[
        (bat_merged["pressure_balls"] >= BATTER_MIN_PRESSURE_BALLS) &
        (bat_merged["pressure_innings"] >= BATTER_MIN_PRESSURE_INNINGS)
    ]["striker_id"].tolist()

    # Bowlers
    bowl_counts = df_p.groupby("bowler_id").agg(
        pressure_balls=("bowler_runs_conceded", "count"),
    ).reset_index()
    bowl_spells = df_p.groupby(["bowler_id", "match_id"]).size().reset_index(name="b")
    bowl_qual_spells = bowl_spells[bowl_spells["b"] >= 6].groupby("bowler_id").size().reset_index(name="pressure_spells")
    bowl_merged = bowl_counts.merge(bowl_qual_spells, on="bowler_id", how="left").fillna(0)

    eligible_bowlers = bowl_merged[
        (bowl_merged["pressure_balls"] >= BOWLER_MIN_PRESSURE_BALLS) &
        (bowl_merged["pressure_spells"] >= BOWLER_MIN_PRESSURE_SPELLS)
    ]["bowler_id"].tolist()

    bat_summary_counts = {"eligible": len(eligible_batters), "total": df["striker_id"].nunique()}
    bowl_summary_counts = {"eligible": len(eligible_bowlers), "total": df["bowler_id"].nunique()}
    print(f"Eligible Batters: {len(eligible_batters)} (minimum {BATTER_MIN_PRESSURE_BALLS} balls, {BATTER_MIN_PRESSURE_INNINGS} innings)")
    print(f"Eligible Bowlers: {len(eligible_bowlers)} (minimum {BOWLER_MIN_PRESSURE_BALLS} deliveries, {BOWLER_MIN_PRESSURE_SPELLS} spells)")

    # 10. Joint Ratings and Consistency
    t0 = time.time()
    print("\n[Step 10/12] Fitting joint regularised ratings (P, N, G) and consistency (C)...")
    bat_ratings, bowl_ratings, diag = fit_player_ratings(df, eligible_batters, eligible_bowlers)
    bat_c, bowl_c = compute_player_consistency(df, eligible_batters, eligible_bowlers)

    bat_base = compute_composite_score(bat_ratings, bat_c)
    bowl_base = compute_composite_score(bowl_ratings, bowl_c)

    print("Computing match-cluster bootstrap confidence intervals (300 replicates)...")
    intervals_bat, intervals_bowl = compute_match_cluster_bootstrap(
        df, eligible_batters, eligible_bowlers, alpha=diag["alpha_pressure"], n_bootstraps=300
    )

    print("Evaluating split-half and temporal reliability...")
    reliability_metrics = run_reliability_checks(df, eligible_batters, eligible_bowlers, n_splits=200)
    print(f"Split-Half Reliability: Batters R = {reliability_metrics['split_half_reliability_bat']:.2f}, Bowlers R = {reliability_metrics['split_half_reliability_bowl']:.2f}")
    print(f"Temporal Check (Pre-2025 vs 2025-2026): Batters rho = {reliability_metrics['temporal_correlation_bat']:.2f}, Bowlers rho = {reliability_metrics['temporal_correlation_bowl']:.2f}")

    # 11. Consensus Bootstrap Ranking
    t0 = time.time()
    print("\n[Step 11/12] Executing 400 consensus bootstrap ranking replicates...")
    bat_cons, bowl_cons = run_consensus_bootstrap(
        df, eligible_batters, eligible_bowlers, alpha=diag["alpha_pressure"], n_replicates=400
    )

    bat_final, bowl_final, final_csv = format_final_rankings(
        bat_base, bat_cons, bowl_base, bowl_cons, datasets["mapping"], intervals_bat, intervals_bowl, df
    )
    print(f"Consensus ranking generated in {time.time() - t0:.2f}s. Saved to outputs/final_rankings.csv")

    # 12. Robustness, Figures and Tables Export
    t0 = time.time()
    print("\n[Step 12/12] Generating publication figures, analytical tables and sensitivity report...")
    top10_bat_ids = bat_final.iloc[:10]["player_id"].tolist()
    top10_bowl_ids = bowl_final.iloc[:10]["player_id"].tolist()

    t8_sensitivity = evaluate_sensitivity_scenarios(
        df, top10_bat_ids, top10_bowl_ids, eligible_batters, eligible_bowlers, alpha=diag["alpha_pressure"]
    )
    run_based_diag = compare_with_run_based_metrics(df, bat_final.iloc[:25], bowl_final.iloc[:25])

    export_all_tables(
        t1_audit,
        wp_metrics,
        outcome_metrics,
        t3_tiers,
        bat_summary_counts,
        bowl_summary_counts,
        bat_final,
        bowl_final,
        t8_sensitivity,
    )
    generate_all_figures(
        df,
        wp_metrics,
        pressure_metrics,
        bat_final,
        bowl_final,
        intervals_bat,
        intervals_bowl,
        reliability_metrics,
    )
    print(f"Figures and tables exported in {time.time() - t0:.2f}s")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY IN {total_time / 60.0:.2f} MINUTES")
    print("=" * 70)

    # Print Official Final Submission Block
    print("\n" + "#" * 70)
    print("OFFICIAL SUBMISSION BLOCK: TOP 10 PLAYERS UNDER PRESSURE")
    print("#" * 70)

    print("\nTOP 10 BATTERS UNDER PRESSURE (Drop-down Selection Order):")
    for _, r in bat_final.iloc[:10].iterrows():
        print(f"Rank {int(r['rank'])}: {r['player_name']} (ID: {int(r['player_id'])}, Team: {r['team_name']})")

    print("\nTOP 10 BOWLERS UNDER PRESSURE (Drop-down Selection Order):")
    for _, r in bowl_final.iloc[:10].iterrows():
        print(f"Rank {int(r['rank'])}: {r['player_name']} (ID: {int(r['player_id'])}, Team: {r['team_name']})")
    print("#" * 70)


if __name__ == "__main__":
    main()
