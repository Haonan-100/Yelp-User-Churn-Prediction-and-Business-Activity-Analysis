# This script consolidates the exploratory steps from the original
# Jupyter notebook (1_data_exploration.ipynb) so they can be executed as
# a standalone Python module.
# ---------------------------------------------------------------
# 0 | Imports & constants
from pathlib import Path
import pandas as pd

RAW = Path("../data/raw")
assert RAW.exists(), "`data/raw` folder not found – check project structure."

# --- Cell 2 ---
# 1 | File row counts (sanity check)
# Quickly count newline characters – inexpensive and avoids loading full JSON.
def count_lines(path: Path, buf_size: int = 1024 * 1024) -> int:
    """Return the number of newline-delimited records in *path*."""
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            n += chunk.count(b"\n")
    return n

FILES = [
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_user.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_tip.json",
    "yelp_academic_dataset_checkin.json",
]

row_counts = {fname: count_lines(RAW / fname) for fname in FILES}
print("Row counts by file:")
for k, v in row_counts.items():
    print(f"  {k}: {v:,}")

# --- Cell 3 ---
# 2 | Review date range & project anchor date
min_d, max_d = None, None
for chunk in pd.read_json(
    RAW / "yelp_academic_dataset_review.json",
    lines=True,
    chunksize=200_000,
    dtype={"user_id": "object", "business_id": "object"},
    usecols=["date"],
):
    if min_d is None:
        min_d = chunk["date"].min()
        max_d = chunk["date"].max()
    else:
        min_d = min(min_d, chunk["date"].min())
        max_d = max(max_d, chunk["date"].max())
print(f"Review dates span from {min_d} to {max_d}")

# --- Cell 4 ---
# 3 | Simple sanity assertion on business & review linkage
print("\nVerifying every review maps to an existing business …")

# Load small sample of business_ids
a_sample = pd.read_json(
    RAW / "yelp_academic_dataset_business.json", lines=True, nrows=5000
)["business_id"].tolist()

b_sample = (
    pd.read_json(
        RAW / "yelp_academic_dataset_review.json",
        lines=True,
        nrows=10_000,
        usecols=["business_id"],
    )["business_id"]
    .isin(a_sample)
    .all()
)
assert b_sample, "Found reviews without matching business_id (sample check)."
print("All sampled reviews have matching business records.")

# --- Cell 5 ---
# 4 | Quick glimpse of business schema
biz_cols = (
    pd.read_json(RAW / "yelp_academic_dataset_business.json", lines=True, nrows=5)
    .columns.tolist()
)
print("\nBusiness file columns:")
for c in biz_cols:
    print("  •", c)

# --- Cell 6 ---
# 5 | Define project anchor (T_END) = latest review date rounded to month end
T_END = pd.to_datetime(max_d).to_period("M").to_timestamp("M")
print("\nProject anchor date (T_END):", T_END.date())

# --- Cell 7 ---
# 6 | Script entry point
if __name__ == "__main__":
    # Running as a script produces the above console outputs.
    pass
