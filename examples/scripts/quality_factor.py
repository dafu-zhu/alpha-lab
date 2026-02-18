"""Quality factor alpha using fundamentals."""
from alphalab.api.client import AlphaLabClient
import os

def quality_alpha(client):
    """ROA-based quality factor, sector-neutralized."""
    # Use pre-computed return_assets field (ROA)
    return client.query("group_neutralize(rank(return_assets), sector)")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = quality_alpha(client)
    print(alpha.tail(10))
