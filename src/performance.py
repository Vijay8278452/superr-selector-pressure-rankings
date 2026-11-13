"""Performance attribution module for the SupeRR pressure analysis pipeline.

This module computes Win Probability Added Above Average (WPAA) and neutral value
per ball (V = WPAA / LI) for batters and bowlers, separating execution quality from
situational leverage. It also computes run-based and wicket-based metrics.
"""

import numpy as np
import pandas as pd


def compute_performance_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute WPAA and neutral value per ball for batters and bowlers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with out-of-fold wp_before, wp_after, expected_wp_change and LI.

    Returns
    -------
    pd.DataFrame
        Dataframe with performance attribution columns.
    """
    df = df.copy()

    # Batting side WPAA: (actual WP change) - (expected WP change)
    # This reflects whether the batting side outperformed an average player's expectation
    wpaa_delivery = df["actual_wp_change"] - df["expected_wp_change"]

    # Attribution rule:
    # Batters receive credit on legal deliveries only
    df["wpaa_bat"] = np.where(df["is_legal"], wpaa_delivery, 0.0)

    # Bowlers receive credit on every delivery bowled (including wides and no-balls)
    # Bowler's gain is the inverse of the batting side's gain
    df["wpaa_bowl"] = -wpaa_delivery

    # Neutral value per ball: V = WPAA / LI
    # Divides by Leverage Index to measure execution quality per unit of pressure
    # A floor of 0.1 on LI avoids numerical instability on rare zero-swing deliveries
    safe_li = np.maximum(0.10, df["leverage_index"])
    df["value_bat"] = df["wpaa_bat"] / safe_li
    df["value_bowl"] = df["wpaa_bowl"] / safe_li

    # Run-based and wicket-based metrics for descriptive comparisons
    df["runs_above_expected"] = np.where(df["is_legal"], df["bat_runs"] - df["expected_runs"], 0.0)
    df["wickets_above_expected"] = df["is_bowler_wicket"].astype(float) - df["expected_wicket"]

    # Express values in standard friendly units: win probability points per 100 balls
    df["value_bat_scaled"] = df["value_bat"] * 100.0
    df["value_bowl_scaled"] = df["value_bowl"] * 100.0

    return df
