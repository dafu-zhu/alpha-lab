"""Mean reversion alpha."""
from alphalab.api.client import AlphaLabClient
import os

from dotenv import load_dotenv
load_dotenv()

def mean_reversion_alpha(client, lookback=20):
    """Buy oversold, sell overbought (z-score based)."""
    return client.query(f"rank(-ts_zscore(close, {lookback}))")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = mean_reversion_alpha(client)
    print(alpha.tail(10))
