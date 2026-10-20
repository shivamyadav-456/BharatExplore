"""Utility functions for logging, JSON reporting, environment checks, and text processing."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting Path, datetime, and numpy types."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_logger(name: str = "BharatExplore") -> logging.Logger:
    """Configures and returns a consistent logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_json_report(data: Dict[str, Any], file_path: Path) -> None:
    """Saves dictionary as formatted JSON report."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=CustomJSONEncoder)


def load_json_report(file_path: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON report if it exists."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_system_environment() -> Dict[str, Any]:
    """Inspects Python, Java, and platform execution context."""
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "java_version": None,
        "java_home": os.environ.get("JAVA_HOME"),
        "java_available": False,
    }
    try:
        res = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        raw = res.stderr or res.stdout
        env_info["java_version"] = raw.splitlines()[0] if raw else "Unknown"
        env_info["java_available"] = True
    except Exception as e:
        env_info["java_error"] = str(e)
    return env_info


def clean_text_for_nlp(text: Optional[str]) -> str:
    """Normalizes text for NLP processing: lowercases, strips punctuation, collapses whitespace.

    Punctuation is removed (not kept) so that lexicon lookups in sentiment_analysis.py
    correctly match dictionary keys (e.g., 'dirty.' -> 'dirty', 'breathtaking,' -> 'breathtaking').
    """
    if not text or not isinstance(text, str):
        return ""
    # Lowercase and normalize whitespace
    t = text.lower().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Remove URLs
    t = re.sub(r"http\S+|www\S+", " ", t)
    # Keep only letters, digits, and whitespace — strip all punctuation so lexicon keys match
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # Collapse multiple whitespaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def calculate_bayesian_rating(
    rating: float,
    review_count: int,
    mean_rating: float,
    min_reviews_threshold: int = 30
) -> float:
    """
    Calculates Bayesian weighted average rating (IMDb-style formula):
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
      v = review count for destination
      m = minimum review threshold
      R = observed average rating
      C = global mean rating across all destinations
    """
    v = float(review_count)
    m = float(min_reviews_threshold)
    return round((v / (v + m)) * rating + (m / (v + m)) * mean_rating, 3)
