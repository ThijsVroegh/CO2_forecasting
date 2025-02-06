from pathlib import Path

# Create base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FORECAST_DIR = DATA_DIR / "forecasts"
HISTORICAL_DIR = DATA_DIR / "historical"
DOWNLOADED_DIR = DATA_DIR / "downloaded"  # New directory for downloaded data

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, FORECAST_DIR, HISTORICAL_DIR, DOWNLOADED_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Training configuration
TRAINING_DAYS = 180  # Set to number of days (e.g., 180) or None for all data 