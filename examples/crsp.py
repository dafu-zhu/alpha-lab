import os
import pandas as pd
import wrds
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("WRDS_USERNAME")
password = os.getenv("WRDS_PASSWORD")

if not username:
    raise ValueError("Missing WRDS_USERNAME in environment/.env")
if not password:
    raise ValueError("Missing WRDS_PASSWORD in environment/.env")

# Many WRDS setups use PGPASSWORD (works across wrds versions)
os.environ["PGPASSWORD"] = password

db = wrds.Connection(wrds_username=username)

print(db.list_libraries())
print(db.describe_table(library="crsp_a_stock", table="dsenames"))

asof = "2009-12-31"

# sql = f"""
# SELECT DISTINCT
#     ticker, permno, comnam, shrcd, exchcd
# FROM crsp_a_stock.dsenames
# WHERE namedt <= '{asof}'
#   AND nameendt >= '{asof}'
#   AND ticker IS NOT NULL
#   AND shrcd IN (10, 11)
#   AND exchcd IN (1, 2, 3)
# ORDER BY ticker;
# """

sql = """
SELECT gvkey, cik, conm, fyear, at, ni
FROM comp.funda
WHERE cik IS NOT NULL;
"""

tickers = db.raw_sql(sql)


print(tickers.head())
print(len(tickers))
