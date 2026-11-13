"""Configuration module for the SupeRR pressure analysis pipeline.

This module defines all constants, file paths, model hyperparameters and
evaluation thresholds used across the pipeline. Centralising these choices
ensures that every downstream module references identical settings.
"""

from pathlib import Path

# Random seed for strict reproducibility across all models and sampling
RANDOM_SEED = 42

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Resolve data directory: first check local data folder, then parent directory
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
PARENT_DATA_DIR = PROJECT_ROOT.parent

if (LOCAL_DATA_DIR / "ball_by_ball.csv").exists():
    DATA_DIR = LOCAL_DATA_DIR
else:
    DATA_DIR = PARENT_DATA_DIR

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"
FINAL_RANKINGS_PATH = OUTPUTS_DIR / "final_rankings.csv"

# Data file names
BALL_BY_BALL_FILE = "ball_by_ball.csv"
MATCHES_FILE = "matches.csv"
COMPETITIONS_FILE = "competitions.csv"
MAPPING_FILE = "competition_team_player_mapping.csv"
PLAYING_ELEVEN_FILE = "playing_eleven.csv"
TEAMS_FILE = "teams.csv"

# Cleaning and filtering rules
EXCLUDED_INNINGS = (3, 4)  # Super overs are analysed separately or excluded
EXCLUDED_RESULTS = ("No Result", "Match Abandoned", "Match Tied")
LEGAL_BALL_EXTRAS_EXCLUDED = ("wide", "Overstep", "other", "Penalty")
BOWLER_CREDITED_DISMISSALS = ("Caught", "Bowled", "LBW", "Stump", "Hit Wicket")

# Cross validation settings
CV_FOLDS = 5

# Win probability model hyperparameters
WP_MODEL_PARAMS = {
    "max_iter": 120,
    "min_samples_leaf": 200,
    "learning_rate": 0.08,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
}

# Ball outcome multiclass model hyperparameters
OUTCOME_MODEL_PARAMS = {
    "max_iter": 100,
    "min_samples_leaf": 150,
    "learning_rate": 0.08,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
}

# Legal ball outcome categories
OUTCOME_CLASSES = ("0", "1", "2", "3", "4", "6", "wicket")

# Leverage index tier cut-offs (percentiles)
LI_TIER_LOW_MAX = 0.40
LI_TIER_MED_MAX = 0.75
LI_TIER_HIGH_MAX = 0.95
# High plus Extreme represents the top 25 percent of all legal deliveries
PRIMARY_PRESSURE_PERCENTILE = 0.75

# Player eligibility cut-offs
BATTER_MIN_PRESSURE_BALLS = 100
BATTER_MIN_PRESSURE_INNINGS = 15
BOWLER_MIN_PRESSURE_BALLS = 90
BOWLER_MIN_PRESSURE_SPELLS = 15

# Pressure innings and spell threshold (deliveries in a single match)
MIN_BALLS_PER_PRESSURE_SPELL = 6

# Consistency score settings
PRIOR_INNINGS_WEIGHT = 10.0  # Beta prior weight for hit rate shrinkage

# Composite score weights
WEIGHT_PRESSURE_RATING = 0.50
WEIGHT_CLUTCH_GAP = 0.20
WEIGHT_CONSISTENCY = 0.30

# Bootstrap and consensus parameters
BOOTSTRAP_REPLICATES = 400
SENSITIVITY_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)
