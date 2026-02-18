"""Combined multi-factor alpha."""
from alphalab.api.client import AlphaLabClient
import os

def combined_alpha(client):
    """Combine momentum, value, and quality signals."""
    return client.query("""
momentum = rank(-ts_delta(close, 20));
value = rank(book / close);
quality = rank(income / assets);
group_neutralize(momentum + value + quality, sector)
""")

if __name__ == "__main__":
    client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])
    alpha = combined_alpha(client)
    print(alpha.tail(10))
