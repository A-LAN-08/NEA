
import os

# Find directories of main python files and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Define directories of data access files
PROFILE_DIR = os.path.join(ROOT_DIR, "profiles")
IMG_DIR = os.path.join(ROOT_DIR, "imgs")
ICON_DIR = os.path.join(PROFILE_DIR, "profile_icons")
CACHE_DIR = os.path.join(ROOT_DIR, "cache_files")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LEDGER_DIR = os.path.join(ROOT_DIR, "ledgers")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Create them if they don't exist
for path in [PROFILE_DIR, IMG_DIR, ICON_DIR, CACHE_DIR, MODEL_DIR, LEDGER_DIR, DATA_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)
