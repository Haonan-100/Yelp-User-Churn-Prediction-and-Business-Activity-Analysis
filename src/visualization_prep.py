"""
Business Analysis + Visualization Prep Pipeline
===============================================
Merged script converted from:
  • 4_business_analysis.ipynb
  • 5_visual_prep.ipynb

Outputs (saved to `data/processed/`)
-----------------------------------
* business_churn_by_category.csv
* state_business_churn.csv
* business_churn_by_open_year.csv
* business_churn_timeline.csv
* user_activity_retention.csv
* user_churn_by_state.csv

Run from project root:
    python -m src.business_visual_prep
"""

# 0 | Imports & global paths --------------------------------------------------
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

RAW  = Path("../data/raw")
PROC = Path("../data/processed"); PROC.mkdir(exist_ok=True, parents=True)

# Anchor dates (should match earlier scripts)
T_END       = pd.Timestamp("2022-01-19")        # last data pull date used previously
CHURN_GAP   = pd.Timedelta(days=365)             # 12‑month churn window
CHURN_CUT   = T_END - CHURN_GAP

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _read_json_iter(path: Path, usecols=None, chunksize=500_000):
    """Chunked JSON reader (line‑delimited)."""
    return pd.read_json(path, lines=True, usecols=usecols, chunksize=chunksize)


def _extract_main_category(cat_str: str) -> str:
    """Return first category token (before comma)."""
    if pd.isna(cat_str):
        return "Unknown"
    return cat_str.split(",")[0].strip()


# -----------------------------------------------------------------------------
# 1 | BUSINESS‑LEVEL CHURN METRICS
# -----------------------------------------------------------------------------

print("🔹 Building business‑level churn table …")

biz_cols = [
    "business_id", "state", "categories", "is_open", "review_count", "stars",
    "open_date"  # some schema have this – fallback if missing later
]

# --- 1.1 last review per business ------------------------------------------------
print("  • scanning reviews to find last review date per business …")
last_review = {}
for chunk in tqdm(
    _read_json_iter(RAW / "yelp_academic_dataset_review.json", usecols=["business_id", "date"]),
    unit="chunk",
):
    grp = chunk.groupby("business_id")["date"].max()
    last_review.update(grp.to_dict())

biz_last_df = (
    pd.Series(last_review, name="last_review_date")
      .rename_axis("business_id")
      .to_frame()
      .reset_index()
)

biz_last_df["churned"] = (biz_last_df["last_review_date"] < CHURN_CUT).astype(int)

# --- 1.2 merge static business fields -------------------------------------------
print("  • reading business.json for static attributes …")
all_biz_df = (
    pd.read_json(RAW / "yelp_academic_dataset_business.json", lines=True, chunksize=1_000_000)
    .pipe(lambda itr: pd.concat(itr, ignore_index=True))
)

if "open_date" not in all_biz_df.columns:
    # synthesize from "years" in review timeline if absent
    all_biz_df["open_date"] = pd.NaT

biz_df = biz_last_df.merge(all_biz_df[biz_cols], on="business_id", how="left")

biz_df["main_cat"] = biz_df["categories"].apply(_extract_main_category)

# Fallbacks
biz_df["state"].fillna("XX", inplace=True)

# -----------------------------------------------------------------------------
# 1-A | Churn by category
cat_grp = biz_df.groupby("main_cat").agg(
    total_business=("business_id", "size"),
    churned=("churned", "sum"),
)
cat_grp["churn_rate"] = cat_grp["churned"] / cat_grp["total_business"]
cat_grp.reset_index().to_csv(PROC / "business_churn_by_category.csv", index=False)

# 1-B | Churn by state
state_grp = biz_df.groupby("state").agg(
    total_business=("business_id", "size"),
    churned=("churned", "sum"),
)
state_grp["churn_rate"] = state_grp["churned"] / state_grp["total_business"]
state_grp.reset_index().to_csv(PROC / "state_business_churn.csv", index=False)

# 1-C | Churn by opening year (if open_date available)
print("  • deriving opening year …")
if biz_df["open_date"].notna().any():
    biz_df["open_year"] = pd.to_datetime(biz_df["open_date"], errors="coerce").dt.year
else:
    # fallback: use first review year
    first_review = {}
    for chunk in tqdm(
        _read_json_iter(RAW / "yelp_academic_dataset_review.json", usecols=["business_id", "date"]),
        desc="    scanning first review dates",
    ):
        grp = chunk.groupby("business_id")["date"].min()
        first_review.update(grp.to_dict())
    first_df = pd.Series(first_review, name="first_review_date").rename_axis("business_id").to_frame()
    biz_df = biz_df.merge(first_df, on="business_id", how="left")
    biz_df["open_year"] = pd.to_datetime(biz_df["first_review_date"], errors="coerce").dt.year

by_year = biz_df.dropna(subset=["open_year"]).groupby("open_year").agg(
    total_business=("business_id", "size"),
    churned=("churned", "sum"),
)
by_year["churn_rate"] = by_year["churned"] / by_year["total_business"]
by_year.reset_index().to_csv(PROC / "business_churn_by_open_year.csv", index=False)

# 1-D | Heatmap timeline (last review year‑month)
print("  • building business churn timeline …")
chron = biz_df.copy()
chron["ym"] = pd.to_datetime(chron["last_review_date"]).dt.to_period("M").astype(str)
chron_grp = chron.groupby("ym").agg(count=("business_id", "size"))
chron_grp.reset_index().to_csv(PROC / "business_churn_timeline.csv", index=False)

# -----------------------------------------------------------------------------
# 2 | USER RETENTION METRICS (reuse earlier churn label) ----------------------
print("🔹 Computing user retention metrics …")

feat_path = PROC / "user_churn_features.parquet"
if feat_path.exists():
    ufeat = pd.read_parquet(feat_path, columns=["user_id", "churn_label", "review_count"])

    # 2‑A | Activity vs retention
    bins = pd.qcut(ufeat["review_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    act_grp = (
        ufeat.assign(activity_bin=bins)
        .groupby("activity_bin")
        .agg(users=("user_id", "size"), retained=("churn_label", lambda x: (1 - x).sum()))
    )
    act_grp["retention_rate"] = act_grp["retained"] / act_grp["users"]
    act_grp.reset_index().to_csv(PROC / "user_activity_retention.csv", index=False)

    # 2‑B | User churn by state (requires mapping user→state from reviews)
    print("  • deriving user home state from most‑reviewed state …")
    user_state = {}
    for chunk in tqdm(
        _read_json_iter(
            RAW / "yelp_academic_dataset_review.json",
            usecols=["user_id", "business_id"],
            chunksize=1_000_000,
        ),
        desc="    scanning user states",
    ):
        chunk = chunk.merge(biz_df[["business_id", "state"]], on="business_id", how="left")
        grp = chunk.groupby(["user_id", "state"]).size().reset_index(name="n")
        top = grp.sort_values("n", ascending=False).drop_duplicates("user_id")
        user_state.update(dict(zip(top["user_id"], top["state"])))
    ufeat["state"] = ufeat["user_id"].map(user_state).fillna("XX")

    state_u = ufeat.groupby("state").agg(
        users=("user_id", "size"),
        churned=("churn_label", "sum"),
    )
    state_u["churn_rate"] = state_u["churned"] / state_u["users"]
    state_u.reset_index().to_csv(PROC / "user_churn_by_state.csv", index=False)
else:
    print("⚠️  user_churn_features.parquet not found – skipping user metrics.")

# -----------------------------------------------------------------------------
print("✅ All exports written to", PROC.resolve())

if __name__ == "__main__":
    pass
