"""Pull the LendingClub loan-level dataset from Kaggle into data/.

Credentials are read from environment variables (KAGGLE_USERNAME, KAGGLE_KEY),
loaded from .env via python-dotenv. Kaggle's own client reads these same
env vars, so no kaggle.json credentials file is written to disk.
"""
from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402  (import after env vars are loaded)

DATASET = "wordsforthewise/lending-club"
DEST = "data"

api = KaggleApi()
api.authenticate()
print(f"Downloading {DATASET} into {DEST}/ ...")
api.dataset_download_files(DATASET, path=DEST, unzip=True, quiet=False)
print("Done.")
