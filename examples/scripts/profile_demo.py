"""Demo: Operator profiling with real data."""
from dotenv import load_dotenv
load_dotenv()

import os
from alphalab.api.client import AlphaLabClient
from alphalab.api.profiler import profile

client = AlphaLabClient(data_path=os.environ["LOCAL_STORAGE_PATH"])

print("Profiling: rank(-ts_delta(close, 5))")
print()

with profile():
    result = client.query("rank(-ts_delta(close, 5))")

print()
print(f"Result shape: {result.shape}")
