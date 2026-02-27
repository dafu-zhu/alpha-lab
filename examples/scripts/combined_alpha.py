"""Combined multi-factor alpha."""
from alphalab.api.client import AlphaLabClient
import os

from dotenv import load_dotenv
load_dotenv()

def combined_alpha(client):
    """Combine momentum and quality signals."""
    import polars as pl
    # Pre-computed momentum (5-day)
    momentum = client.query("rank(-ts_delta(close, 5))")
    # Pre-computed quality (ROA)
    quality = client.query("rank(return_assets)")
    # Find common columns (excluding Date)
    common_cols = sorted(set(momentum.columns) & set(quality.columns) - {"Date"})
    # Combine on common securities
    mom_vals = momentum.select(["Date"] + common_cols)
    qual_vals = quality.select(["Date"] + common_cols)
    # Add numeric columns only
    combined = mom_vals.select("Date").hstack([
        (mom_vals[c] + qual_vals[c]).alias(c) for c in common_cols
    ])
    return combined

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = combined_alpha(client)
    print(alpha.tail(10))
