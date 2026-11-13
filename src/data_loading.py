"""Data loading module for the SupeRR pressure analysis pipeline.

This module loads all raw CSV files from the configured data directory,
validates file presence and provides unified access for cleaning and modelling.
"""

from pathlib import Path
from typing import Dict
import pandas as pd

from src.config import (
    DATA_DIR,
    BALL_BY_BALL_FILE,
    MATCHES_FILE,
    COMPETITIONS_FILE,
    MAPPING_FILE,
    PLAYING_ELEVEN_FILE,
    TEAMS_FILE,
)


def load_raw_datasets(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV datasets into pandas dataframes.

    Parameters
    ----------
    data_dir : Path
        Directory where CSV files are stored.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to loaded dataframes.
    """
    file_map = {
        "ball_by_ball": BALL_BY_BALL_FILE,
        "matches": MATCHES_FILE,
        "competitions": COMPETITIONS_FILE,
        "mapping": MAPPING_FILE,
        "playing_eleven": PLAYING_ELEVEN_FILE,
        "teams": TEAMS_FILE,
    }

    datasets = {}
    for key, filename in file_map.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required data file not found: {filepath}")

        # Use low_memory=False for ball_by_ball to avoid mixed type warnings
        if key == "ball_by_ball":
            df = pd.read_csv(filepath, low_memory=False)
        else:
            df = pd.read_csv(filepath)

        datasets[key] = df

    return datasets
