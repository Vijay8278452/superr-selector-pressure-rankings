"""Table generation module for the SupeRR pressure analysis pipeline.

This module formats and saves all 8 analytical tables (T1 through T8) in both
machine readable CSV and publication ready LaTeX formats. Table text and headers
strictly follow the style rules: no em dashes, no en dashes, and no Oxford commas.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.config import TABLES_DIR


def dataframe_to_latex_snippet(df: pd.DataFrame, col_align: str = None) -> str:
    """Convert dataframe to clean LaTeX tabular snippet using booktabs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert.
    col_align : str, optional
        Column alignment string (e.g. 'llrrrr').

    Returns
    -------
    str
        LaTeX tabular string.
    """
    cols = df.columns.tolist()
    if col_align is None:
        col_align = "l" * len(cols)

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\toprule")
    # Clean headers
    headers = [str(c).replace("_", " ").replace("%", "\\%") for c in cols]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        row_vals = []
        for val in row:
            val_str = str(val).replace("_", "\\_").replace("%", "\\%")
            row_vals.append(val_str)
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def save_table_csv_and_tex(
    df: pd.DataFrame,
    filename_base: str,
    caption: str,
    label: str,
    output_dir: Path = TABLES_DIR,
):
    """Save dataframe as CSV and ready-to-include LaTeX table snippet.

    Parameters
    ----------
    df : pd.DataFrame
        Table data.
    filename_base : str
        Base filename without extension (e.g. 'table_1_data_audit').
    caption : str
        Table caption for LaTeX document.
    label : str
        Table label for LaTeX referencing.
    output_dir : Path
        Directory where tables are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    csv_path = output_dir / f"{filename_base}.csv"
    df.to_csv(csv_path, index=False)

    # 2. Save LaTeX snippet with booktabs styling
    tex_path = output_dir / f"{filename_base}.tex"
    tabular_content = dataframe_to_latex_snippet(df)

    full_tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\resizebox{{\\textwidth}}{{!}}{{%
{tabular_content}
}}
\\end{{table}}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(full_tex)


def export_all_tables(
    t1_audit: pd.DataFrame,
    wp_metrics: Dict[str, any],
    outcome_metrics: Dict[str, any],
    t3_tiers: pd.DataFrame,
    bat_counts_summary: Dict[str, int],
    bowl_counts_summary: Dict[str, int],
    bat_top: pd.DataFrame,
    bowl_top: pd.DataFrame,
    t8_sensitivity: pd.DataFrame,
    output_dir: Path = TABLES_DIR,
):
    """Generate all tables T1 to T8 in CSV and LaTeX format.

    Parameters
    ----------
    t1_audit : pd.DataFrame
        Data audit table.
    wp_metrics : Dict[str, any]
        Win probability metrics.
    outcome_metrics : Dict[str, any]
        Outcome multiclass model metrics.
    t3_tiers : pd.DataFrame
        Tier outcomes table.
    bat_counts_summary : Dict[str, int]
        Batting sample size summary.
    bowl_counts_summary : Dict[str, int]
        Bowling sample size summary.
    bat_top : pd.DataFrame
        Full ranked batters dataframe.
    bowl_top : pd.DataFrame
        Full ranked bowlers dataframe.
    t8_sensitivity : pd.DataFrame
        Sensitivity scenarios dataframe.
    output_dir : Path
        Output directory.
    """
    # Table 1: Data Audit
    save_table_csv_and_tex(
        t1_audit,
        "table_1_data_audit",
        "Data audit, verification and sample exclusion counts",
        "tab:data_audit",
        output_dir,
    )

    # Table 2: Model Performance Metrics
    t2_records = [
        {"Model Component": "Innings 2 Chase WP Model (HGB)", "Out of Fold Log Loss": f"{wp_metrics['inn2_hgb_logloss']:.4f}", "ROC AUC": f"{wp_metrics['inn2_hgb_auc']:.4f}", "Brier Score": f"{wp_metrics['inn2_hgb_brier']:.4f}", "Baseline Log Loss": f"{wp_metrics['inn2_lr_logloss']:.4f} (LR)"},
        {"Model Component": "Innings 1 Setting WP Model (HGB)", "Out of Fold Log Loss": f"{wp_metrics['inn1_hgb_logloss']:.4f}", "ROC AUC": f"{wp_metrics['inn1_hgb_auc']:.4f}", "Brier Score": f"{wp_metrics['inn1_hgb_brier']:.4f}", "Baseline Log Loss": f"{wp_metrics['inn1_lr_logloss']:.4f} (LR)"},
        {"Model Component": "Ball Outcome Multiclass Model", "Out of Fold Log Loss": f"{outcome_metrics['outcome_model_logloss']:.4f}", "ROC AUC": "N/A (Multiclass)", "Brier Score": "N/A", "Baseline Log Loss": f"{outcome_metrics['naive_phase_logloss']:.4f} (Phase Prior)"},
    ]
    t2_df = pd.DataFrame(t2_records)
    save_table_csv_and_tex(
        t2_df,
        "table_2_model_performance",
        "Out of fold model performance metrics and baseline comparisons",
        "tab:model_performance",
        output_dir,
    )

    # Table 3: Pressure Tiers Outcomes
    save_table_csv_and_tex(
        t3_tiers,
        "table_3_pressure_tiers",
        "League wide execution outcomes across pre-ball leverage index tiers",
        "tab:pressure_tiers",
        output_dir,
    )

    # Table 4: Eligibility Thresholds and Player Counts
    t4_records = [
        {"Role": "Batters", "Minimum Pressure Balls": 100, "Minimum Pressure Innings": 15, "Eligible Players": bat_counts_summary["eligible"], "Total Active Players": bat_counts_summary["total"]},
        {"Role": "Bowlers", "Minimum Pressure Deliveries": 90, "Minimum Pressure Spells": 15, "Eligible Players": bowl_counts_summary["eligible"], "Total Active Players": bowl_counts_summary["total"]},
    ]
    t4_df = pd.DataFrame(t4_records)
    save_table_csv_and_tex(
        t4_df,
        "table_4_eligibility",
        "Sample size eligibility cut-offs and pool sizes",
        "tab:eligibility",
        output_dir,
    )

    # Table 5: Top 10 Batters Under Pressure
    t5_cols = [
        "rank", "player_name", "team_name", "pressure_balls", "pressure_inns",
        "pressure_rating_p", "ci80_lower", "ci80_upper", "normal_rating_n",
        "clutch_gap_g", "consistency_score_c", "composite_score_s", "top10_probability",
        "pressure_strike_rate", "normal_strike_rate", "pressure_dismissal_pct", "dot_pct", "boundary_pct"
    ]
    t5_df = bat_top.iloc[:10][t5_cols].copy()
    t5_df["P [80% CI]"] = t5_df.apply(
        lambda r: f"{r['pressure_rating_p']:.3f} [{r['ci80_lower']:.3f} to {r['ci80_upper']:.3f}]", axis=1
    )
    t5_display = pd.DataFrame({
        "Rank": t5_df["rank"],
        "Player Name": t5_df["player_name"],
        "Team": t5_df["team_name"],
        "Balls": t5_df["pressure_balls"].astype(int),
        "Inns": t5_df["pressure_inns"].astype(int),
        "P [80% CI]": t5_df["P [80% CI]"],
        "N": t5_df["normal_rating_n"].round(3),
        "G": t5_df["clutch_gap_g"].round(3),
        "C": t5_df["consistency_score_c"].round(2),
        "Score S": t5_df["composite_score_s"].round(3),
        "Top 10 Prob": (t5_df["top10_probability"] * 100.0).round(1).astype(str) + "%",
        "Press SR": t5_df["pressure_strike_rate"].round(1),
        "Norm SR": t5_df["normal_strike_rate"].round(1),
        "Dism %": t5_df["pressure_dismissal_pct"].round(1),
        "Dot %": t5_df["dot_pct"].round(1),
        "Bound %": t5_df["boundary_pct"].round(1),
    })
    save_table_csv_and_tex(
        t5_display,
        "table_5_top10_batters",
        "Top 10 batters under pressure with consensus rankings and performance metrics",
        "tab:top10_batters",
        output_dir,
    )

    # Table 6: Top 10 Bowlers Under Pressure
    t6_cols = [
        "rank", "player_name", "team_name", "pressure_deliveries", "pressure_spells",
        "pressure_rating_p", "ci80_lower", "ci80_upper", "normal_rating_n",
        "clutch_gap_g", "consistency_score_c", "composite_score_s", "top10_probability",
        "pressure_economy", "normal_economy", "wicket_rate", "dot_pct", "boundary_conceded_pct", "extras_pct"
    ]
    t6_df = bowl_top.iloc[:10][t6_cols].copy()
    t6_df["P [80% CI]"] = t6_df.apply(
        lambda r: f"{r['pressure_rating_p']:.3f} [{r['ci80_lower']:.3f} to {r['ci80_upper']:.3f}]", axis=1
    )
    t6_display = pd.DataFrame({
        "Rank": t6_df["rank"],
        "Player Name": t6_df["player_name"],
        "Team": t6_df["team_name"],
        "Balls": t6_df["pressure_deliveries"].astype(int),
        "Spells": t6_df["pressure_spells"].astype(int),
        "P [80% CI]": t6_df["P [80% CI]"],
        "N": t6_df["normal_rating_n"].round(3),
        "G": t6_df["clutch_gap_g"].round(3),
        "C": t6_df["consistency_score_c"].round(2),
        "Score S": t6_df["composite_score_s"].round(3),
        "Top 10 Prob": (t6_df["top10_probability"] * 100.0).round(1).astype(str) + "%",
        "Press Econ": t6_df["pressure_economy"].round(2),
        "Norm Econ": t6_df["normal_economy"].round(2),
        "Wkt %": t6_df["wicket_rate"].round(1),
        "Dot %": t6_df["dot_pct"].round(1),
        "Bound %": t6_df["boundary_conceded_pct"].round(1),
        "Extras %": t6_df["extras_pct"].round(1),
    })
    save_table_csv_and_tex(
        t6_display,
        "table_6_top10_bowlers",
        "Top 10 bowlers under pressure with consensus rankings and performance metrics",
        "tab:top10_bowlers",
        output_dir,
    )

    # Table 7: Ranks 11 to 20 for Comparison
    t7_bat = bat_top.iloc[10:20].copy()
    t7_bowl = bowl_top.iloc[10:20].copy()

    t7_display = pd.DataFrame({
        "Rank": range(11, 21),
        "Batter Name": t7_bat["player_name"].values,
        "Batter Team": t7_bat["team_name"].values,
        "Batter P": t7_bat["pressure_rating_p"].round(3).values,
        "Batter Score": t7_bat["composite_score_s"].round(3).values,
        "Bowler Name": t7_bowl["player_name"].values,
        "Bowler Team": t7_bowl["team_name"].values,
        "Bowler P": t7_bowl["pressure_rating_p"].round(3).values,
        "Bowler Score": t7_bowl["composite_score_s"].round(3).values,
    })
    save_table_csv_and_tex(
        t7_display,
        "table_7_ranks_11_to_20",
        "Players ranked 11 to 20 under pressure for comparative reference",
        "tab:ranks_11_to_20",
        output_dir,
    )

    # Table 8: Sensitivity Scenarios
    save_table_csv_and_tex(
        t8_sensitivity,
        "table_8_sensitivity",
        "Sensitivity analysis of top 10 player selections across model assumptions",
        "tab:sensitivity",
        output_dir,
    )
