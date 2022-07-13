
import os

# Cleaned data variables
DATA_DIR = "data"
META_FILE = "metadata.json"
METADATA = os.path.join(DATA_DIR, META_FILE)
RATING_FILE = "rating.csv"
RATING = os.path.join(DATA_DIR, RATING_FILE)
DATABASE = os.path.join(DATA_DIR, "movielens")
TABLE = "metadata"

# Raw data variables
RAW_DIR = "raw"  # raw folders
RAW_META = "movies.dat"  # raw metadata movielens file
RAW_META_FILE = os.path.join(RAW_DIR, RAW_META)  # path to raw metdata
RAW_RATING = "ratings.dat"  # raw rating movielens file
RAW_RATING_FILE = os.path.join(RAW_DIR, RAW_RATING)  # path to rating file

# Model variables
MODEL_DIR = "models"
CB_FILE = "cb.pkl"
CB_MODEL_PATH = os.path.join(MODEL_DIR, CB_FILE)
ALS_FILE = "als.pkl"
ALS_MODEL_PATH = os.path.join(MODEL_DIR, ALS_FILE)

# Rating file columns selected
COLUMNS = ['UserID', "MovieID", "Rating"]



