"""
User Churn Feature Engineering Pipeline
--------------------------------------
Converted from `2_user_churn_features.ipynb` so it can run headless.
Assumes the project data folder structure:
  data/
    raw/        # original Yelp JSON files
    processed/  # output features & split lists

Run this script from project root:
    python -m src.user_churn_features
"""

# --- Cell ---
# 0 | Imports & Global Constants
from pathlib import Path
import pandas as pd
from tqdm import tqdm

RAW   = Path("../data/raw")
PROC  = Path("../data/processed");  PROC.mkdir(exist_ok=True)

T_END        = pd.Timestamp("2022-01-19")   # by part 1
WINDOW_DAYS  = 365
CUTOFF_DATE  = T_END - pd.Timedelta(days=WINDOW_DAYS)

# --- Cell ---

def clean_state(df, col="state"):
    mask = ~df[col].str.match(r"^[A-Z]{2}$", na=False)
    df.loc[mask, col] = "XX"
    return df

# --- Cell ---

def safe_read_json(fname, usecols=None, chunksize=None):
    path = RAW / fname
    if chunksize:
        return pd.read_json(path, lines=True, dtype=dict, usecols=usecols, chunksize=chunksize)
    return pd.read_json(path, lines=True, dtype=dict, usecols=usecols)

# --- Cell ---
# 1 | gey `churn_label`
last_review = {}
for chunk in tqdm(
    safe_read_json(
        "yelp_academic_dataset_review.json",
        usecols=["user_id", "date"],
        chunksize=1_000_000,
    )
):
    chunk = chunk[chunk["date"] >= CUTOFF_DATE]
    grouped = chunk.groupby("user_id")["date"].max()
    last_review.update(grouped.to_dict())

user_last_df = pd.DataFrame.from_dict(last_review, orient="index", columns=["last_review_date"]).reset_index().rename(columns={"index": "user_id"})
user_last_df["churn_label"] = (user_last_df["last_review_date"] < pd.Timestamp("2021-01-19")).astype(int)

# --- Cell ---
# 1.5 Re-label churn based on fixed anchor (T_END)
ANCHOR_CUTOFF = pd.Timestamp("2021-01-19")          # T_END - 365 days

# Already calculated above

# --- Cell ---
# 2 | User Static Features (from `user.json`)
user_cols = [
    "user_id", "review_count", "average_stars",
]
user_df = safe_read_json("yelp_academic_dataset_user.json", usecols=user_cols)

# --- Cell ---
# 3 | Behavioral Features (`review.json`)
behav_chunks = []
for chunk in tqdm(
    safe_read_json(
        "yelp_academic_dataset_review.json",
        usecols=["user_id", "stars"],
        chunksize=1_000_000,
    )
):
    grp = chunk.groupby("user_id").agg(review_mean_stars=("stars", "mean"), review_cnt=("stars", "size"))
    behav_chunks.append(grp)
behav_df = pd.concat(behav_chunks).groupby(level=0).sum().reset_index()

# --- Cell ---
# 4 | Merging & Missing Value Handling
feat_df = (
    user_last_df
    .merge(user_df, on="user_id", how="left")
    .merge(behav_df, on="user_id", how="left")
)

feat_df["review_mean_stars"].fillna(0, inplace=True)
feat_df["review_cnt"].fillna(0, inplace=True)

feat_df.to_parquet(PROC / "user_churn_features.parquet", index=False)

# --- Cell ---
print(feat_df["churn_label"].value_counts(dropna=False))

# --- Cell ---
# 6-A | Use stratified sampling to split the training and testing user IDs
from sklearn.model_selection import train_test_split

train_ids, test_ids = train_test_split(
    feat_df["user_id"], stratify=feat_df["churn_label"], test_size=0.2, random_state=42
)

# --- Cell ---
# 6-B | Save the new training and testing user ID lists
train_ids.to_csv( PROC/"train_user_ids.txt", index=False, header=False)
test_ids.to_csv( PROC/"test_user_ids.txt",  index=False, header=False)

if __name__ == "__main__":
    print(">>> Running user churn feature engineering pipeline...")
    # Code cells executed above during import; nothing else required.
