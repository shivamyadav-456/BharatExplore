"""Export demo data script for BharatExplore.

Exports a small, representative sample of cleaned reviews and destination profiles
to ensure instant offline availability and lightweight unit test demonstration.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ANALYTICS_DATA_DIR,
    DESTINATION_PROFILES_PARQUET,
    PROCESSED_DATA_DIR,
)
from src.utils import get_logger

logger = get_logger("ExportDemoData")


def export_demo_data(num_destinations: int = 200, output_dir: Path = None):
    """Samples destination profiles and exports a lightweight demo set."""
    if output_dir is None:
        output_dir = ANALYTICS_DATA_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not DESTINATION_PROFILES_PARQUET.exists():
        logger.error(f"Cannot export demo data: {DESTINATION_PROFILES_PARQUET} does not exist.")
        return False

    df = pd.read_parquet(DESTINATION_PROFILES_PARQUET)
    sample_df = df.sample(min(num_destinations, len(df)), random_state=42)
    out_file = output_dir / "demo_destinations.parquet"
    sample_df.to_parquet(out_file, index=False)
    logger.info(f"Exported {len(sample_df)} demo destination profiles to {out_file}")
    return True


if __name__ == "__main__":
    export_demo_data()
