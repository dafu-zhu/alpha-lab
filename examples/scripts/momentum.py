"""Simple momentum alpha."""
from alphalab.api.client import AlphaLabClient
import os

def momentum_alpha(client, lookback=5):
    """5-day price momentum, cross-sectionally ranked."""
    return client.query(f"rank(-ts_delta(close, {lookback}))")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = momentum_alpha(client)
    print(alpha.tail(10))
