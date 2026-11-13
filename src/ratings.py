"""Joint player ratings module with regularised shrinkage for the SupeRR pipeline.

This module fits a simultaneous linear model of batters and bowlers with Ridge
regression to adjust for opposition quality. It computes Pressure Ratings (P),
Normal Ratings (N), the Clutch Gap (G = P - N), Empirical Bayes shrinkage cross checks,
and match-clustered bootstrap confidence intervals.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from src.config import RANDOM_SEED, CV_FOLDS


def build_joint_design_matrix(
    df_subset: pd.DataFrame,
    batter_list: List[int],
    bowler_list: List[int],
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Construct sparse delivery level design matrix with opposing effects.

    Batters receive +1.0 in their respective column on legal deliveries.
    Bowlers receive -1.0 in their respective column on all deliveries.

    Parameters
    ----------
    df_subset : pd.DataFrame
        Subset of deliveries to model (pressure or non-pressure).
    batter_list : List[int]
        Ordered list of unique batter player IDs.
    bowler_list : List[int]
        Ordered list of unique bowler player IDs.

    Returns
    -------
    Tuple[sp.csr_matrix, np.ndarray]
        Sparse design matrix X and target vector y (scaled neutral value).
    """
    n_rows = len(df_subset)
    n_bat = len(batter_list)
    n_bowl = len(bowler_list)
    n_cols = n_bat + n_bowl

    batter_map = {pid: i for i, pid in enumerate(batter_list)}
    bowler_map = {pid: n_bat + i for i, pid in enumerate(bowler_list)}

    row_ind: List[int] = []
    col_ind: List[int] = []
    data: List[float] = []

    strikers = df_subset["striker_id"].values
    bowlers = df_subset["bowler_id"].values
    is_legals = df_subset["is_legal"].values

    for r in range(n_rows):
        # Bowler is credited on every delivery (-1 in bowler column)
        bw_id = bowlers[r]
        if bw_id in bowler_map:
            row_ind.append(r)
            col_ind.append(bowler_map[bw_id])
            data.append(-1.0)

        # Batter is credited on legal deliveries only (+1 in batter column)
        if is_legals[r]:
            bt_id = strikers[r]
            if bt_id in batter_map:
                row_ind.append(r)
                col_ind.append(batter_map[bt_id])
                data.append(1.0)

    X = sp.csr_matrix((data, (row_ind, col_ind)), shape=(n_rows, n_cols), dtype=np.float32)
    # Target is scaled neutral value from batting side perspective
    y = df_subset["value_bat_scaled"].values

    return X, y


def tune_ridge_alpha(
    X: sp.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_candidates: List[float] = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
) -> float:
    """Select regularisation penalty alpha via match-grouped cross validation.

    Parameters
    ----------
    X : sp.csr_matrix
        Sparse design matrix.
    y : np.ndarray
        Target vector.
    groups : np.ndarray
        Match ID groups.
    alpha_candidates : List[float]
        Candidate penalty values.

    Returns
    -------
    float
        Optimal alpha minimising mean squared error.
    """
    gkf = GroupKFold(n_splits=CV_FOLDS)
    best_mse = float("inf")
    best_alpha = 100.0

    for alpha in alpha_candidates:
        fold_mses = []
        for trn, val in gkf.split(X, y, groups):
            ridge = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
            ridge.fit(X[trn], y[trn])
            preds = ridge.predict(X[val])
            mse = np.mean((y[val] - preds) ** 2)
            fold_mses.append(mse)

        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha

    return best_alpha


def fit_player_ratings(
    df: pd.DataFrame,
    batter_list: List[int],
    bowler_list: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """Fit joint regularised ratings on pressure balls and non-pressure balls.

    Parameters
    ----------
    df : pd.DataFrame
        Complete delivery dataframe with performance attribution.
    batter_list : List[int]
        Unique eligible or active batter IDs.
    bowler_list : List[int]
        Unique eligible or active bowler IDs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]
        Batting ratings dataframe, bowling ratings dataframe, and model diagnostics.
    """
    df_p = df[df["is_pressure"]].reset_index(drop=True)
    df_n = df[~df["is_pressure"]].reset_index(drop=True)

    X_p, y_p = build_joint_design_matrix(df_p, batter_list, bowler_list)
    groups_p = df_p["match_id"].values
    alpha_p = tune_ridge_alpha(X_p, y_p, groups_p)

    model_p = Ridge(alpha=alpha_p, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
    model_p.fit(X_p, y_p)
    coefs_p = model_p.coef_

    X_n, y_n = build_joint_design_matrix(df_n, batter_list, bowler_list)
    groups_n = df_n["match_id"].values
    alpha_n = tune_ridge_alpha(X_n, y_n, groups_n)

    model_n = Ridge(alpha=alpha_n, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
    model_n.fit(X_n, y_n)
    coefs_n = model_n.coef_

    n_bat = len(batter_list)

    # Batters receive positive coefficients for good performance
    p_bat = coefs_p[:n_bat]
    n_bat_ratings = coefs_n[:n_bat]

    # Bowlers receive positive coefficients for good performance because column sign is -1
    p_bowl = coefs_p[n_bat:]
    n_bowl_ratings = coefs_n[n_bat:]

    bat_df = pd.DataFrame({
        "player_id": batter_list,
        "pressure_rating_p": p_bat,
        "normal_rating_n": n_bat_ratings,
        "clutch_gap_g": p_bat - n_bat_ratings,
    })

    bowl_df = pd.DataFrame({
        "player_id": bowler_list,
        "pressure_rating_p": p_bowl,
        "normal_rating_n": n_bowl_ratings,
        "clutch_gap_g": p_bowl - n_bowl_ratings,
    })

    # Empirical Bayes cross check on pressure subset
    eb_bat = df_p[df_p["is_legal"]].groupby("striker_id")["value_bat_scaled"].agg(["count", "mean", "var"])
    grand_mean_bat = df_p[df_p["is_legal"]]["value_bat_scaled"].mean()
    s2_noise_bat = eb_bat["var"].median()
    s2_talent_bat = max(0.01, eb_bat["mean"].var() - s2_noise_bat / max(1.0, eb_bat["count"].median()))
    k_eb_bat = s2_noise_bat / s2_talent_bat
    eb_bat["eb_shrunk"] = (eb_bat["count"] * eb_bat["mean"] + k_eb_bat * grand_mean_bat) / (eb_bat["count"] + k_eb_bat)
    bat_df["eb_rating"] = bat_df["player_id"].map(eb_bat["eb_shrunk"]).fillna(0.0)

    eb_bowl = df_p.groupby("bowler_id")["value_bowl_scaled"].agg(["count", "mean", "var"])
    grand_mean_bowl = df_p["value_bowl_scaled"].mean()
    s2_noise_bowl = eb_bowl["var"].median()
    s2_talent_bowl = max(0.01, eb_bowl["mean"].var() - s2_noise_bowl / max(1.0, eb_bowl["count"].median()))
    k_eb_bowl = s2_noise_bowl / s2_talent_bowl
    eb_bowl["eb_shrunk"] = (eb_bowl["count"] * eb_bowl["mean"] + k_eb_bowl * grand_mean_bowl) / (eb_bowl["count"] + k_eb_bowl)
    bowl_df["eb_rating"] = bowl_df["player_id"].map(eb_bowl["eb_shrunk"]).fillna(0.0)

    # Rank correlation between Ridge and Empirical Bayes
    rho_bat, _ = spearmanr(bat_df["pressure_rating_p"], bat_df["eb_rating"])
    rho_bowl, _ = spearmanr(bowl_df["pressure_rating_p"], bowl_df["eb_rating"])

    diagnostics = {
        "alpha_pressure": alpha_p,
        "alpha_normal": alpha_n,
        "eb_correlation_bat": float(rho_bat),
        "eb_correlation_bowl": float(rho_bowl),
    }

    return bat_df, bowl_df, diagnostics


def compute_match_cluster_bootstrap(
    df: pd.DataFrame,
    batter_list: List[int],
    bowler_list: List[int],
    alpha: float,
    n_bootstraps: int = 300,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Compute standard errors and confidence intervals by resampling whole matches.

    Parameters
    ----------
    df : pd.DataFrame
        Complete delivery level dataframe.
    batter_list : List[int]
        Unique batter IDs.
    bowler_list : List[int]
        Unique bowler IDs.
    alpha : float
        Regularisation parameter.
    n_bootstraps : int
        Number of bootstrap replications.

    Returns
    -------
    Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]
        Dictionaries of interval metrics for batters and bowlers.
    """
    df_p = df[df["is_pressure"]].reset_index(drop=True)
    unique_matches = df_p["match_id"].unique()
    n_matches = len(unique_matches)

    n_bat = len(batter_list)
    n_bowl = len(bowler_list)

    bat_boot_dist = np.zeros((n_bootstraps, n_bat))
    bowl_boot_dist = np.zeros((n_bootstraps, n_bowl))

    # Pre-group index by match for fast resampling
    match_indices = df_p.groupby("match_id").indices

    rng = np.random.default_rng(RANDOM_SEED)

    for b in range(n_bootstraps):
        sampled_matches = rng.choice(unique_matches, size=n_matches, replace=True)
        sampled_rows = np.concatenate([match_indices[m] for m in sampled_matches])
        boot_sub = df_p.iloc[sampled_rows].reset_index(drop=True)

        X_b, y_b = build_joint_design_matrix(boot_sub, batter_list, bowler_list)
        model_b = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=RANDOM_SEED)
        model_b.fit(X_b, y_b)

        bat_boot_dist[b, :] = model_b.coef_[:n_bat]
        bowl_boot_dist[b, :] = model_b.coef_[n_bat:]

    bat_intervals: Dict[int, Dict[str, float]] = {}
    for i, pid in enumerate(batter_list):
        vals = bat_boot_dist[:, i]
        bat_intervals[pid] = {
            "se": float(np.std(vals)),
            "ci80_lower": float(np.percentile(vals, 10.0)),
            "ci80_upper": float(np.percentile(vals, 90.0)),
            "ci95_lower": float(np.percentile(vals, 2.5)),
            "ci95_upper": float(np.percentile(vals, 97.5)),
        }

    bowl_intervals: Dict[int, Dict[str, float]] = {}
    for i, pid in enumerate(bowler_list):
        vals = bowl_boot_dist[:, i]
        bowl_intervals[pid] = {
            "se": float(np.std(vals)),
            "ci80_lower": float(np.percentile(vals, 10.0)),
            "ci80_upper": float(np.percentile(vals, 90.0)),
            "ci95_lower": float(np.percentile(vals, 2.5)),
            "ci95_upper": float(np.percentile(vals, 97.5)),
        }

    return bat_intervals, bowl_intervals
