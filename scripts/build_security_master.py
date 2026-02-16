"""
Standalone one-time WRDS build script for SecurityMaster.

Builds the security master parquet from WRDS CRSP data with Compustat
exchange + GICS classification. No imports from alphalab.master.

Usage:
    uv run python scripts/build_security_master.py

Output: data/meta/master/security_master.parquet
Schema: security_id, permno, symbol, company, cik,
        start_date, end_date, exchange, sector, industry, subindustry

Requires: wrds package (uv add wrds)
"""

import sys
import os
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv

load_dotenv()


# ── Compustat exchange code → name ──────────────────────────────────────────
EXCHG_MAP = {
    11: "NYSE",
    12: "AMEX",
    14: "NASDAQ",
}

# ── GICS sector code → name ────────────────────────────────────────────────
GICS_SECTOR = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# ── GICS industry group code → name ────────────────────────────────────────
GICS_INDUSTRY_GROUP = {
    "1010": "Energy",
    "1510": "Materials",
    "2010": "Capital Goods",
    "2020": "Commercial & Professional Services",
    "2030": "Transportation",
    "2510": "Automobiles & Components",
    "2520": "Consumer Durables & Apparel",
    "2530": "Consumer Services",
    "2550": "Consumer Discretionary Distribution & Retail",
    "2560": "Consumer Staples Distribution & Retail",
    "3010": "Food, Beverage & Tobacco",
    "3020": "Household & Personal Products",
    "3030": "Food & Staples Retailing",
    "3510": "Health Care Equipment & Services",
    "3520": "Pharmaceuticals, Biotechnology & Life Sciences",
    "4010": "Banks",
    "4020": "Financial Services",
    "4030": "Insurance",
    "4040": "Real Estate",
    "4510": "Software & Services",
    "4520": "Technology Hardware & Equipment",
    "4530": "Semiconductors & Semiconductor Equipment",
    "5010": "Telecommunication Services",
    "5020": "Media & Entertainment",
    "5510": "Utilities",
    "6010": "Equity Real Estate Investment Trusts (REITs)",
    "6020": "Real Estate Management & Development",
}

# ── GICS sub-industry code → name ──────────────────────────────────────────
GICS_SUBINDUSTRY = {
    # Energy
    "10101010": "Oil & Gas Drilling",
    "10101020": "Oil & Gas Equipment & Services",
    "10102010": "Integrated Oil & Gas",
    "10102020": "Oil & Gas Exploration & Production",
    "10102030": "Oil & Gas Refining & Marketing",
    "10102040": "Oil & Gas Storage & Transportation",
    "10102050": "Coal & Consumable Fuels",
    # Materials
    "15101010": "Commodity Chemicals",
    "15101020": "Diversified Chemicals",
    "15101030": "Fertilizers & Agricultural Chemicals",
    "15101040": "Industrial Gases",
    "15101050": "Specialty Chemicals",
    "15102010": "Construction Materials",
    "15103010": "Metal, Glass & Plastic Containers",
    "15103020": "Paper & Plastic Packaging Products & Materials",
    "15104010": "Aluminum",
    "15104020": "Diversified Metals & Mining",
    "15104025": "Copper",
    "15104030": "Gold",
    "15104040": "Precious Metals & Minerals",
    "15104045": "Silver",
    "15104050": "Steel",
    "15105010": "Forest Products",
    "15105020": "Paper Products",
    # Industrials
    "20101010": "Aerospace & Defense",
    "20102010": "Building Products",
    "20103010": "Construction & Engineering",
    "20104010": "Electrical Components & Equipment",
    "20104020": "Heavy Electrical Equipment",
    "20105010": "Industrial Conglomerates",
    "20106010": "Construction Machinery & Heavy Transportation Equipment",
    "20106015": "Agricultural & Farm Machinery",
    "20106020": "Industrial Machinery & Supplies & Components",
    "20107010": "Trading Companies & Distributors",
    "20201010": "Commercial Printing",
    "20201050": "Environmental & Facilities Services",
    "20201060": "Office Services & Supplies",
    "20201070": "Diversified Support Services",
    "20201080": "Security & Alarm Services",
    "20202010": "Human Resource & Employment Services",
    "20202020": "Research & Consulting Services",
    "20203010": "Data Processing & Outsourced Services",
    "20301010": "Air Freight & Logistics",
    "20302010": "Passenger Airlines",
    "20303010": "Marine Transportation",
    "20304010": "Rail Transportation",
    "20304020": "Trucking",
    "20305010": "Airport Services",
    "20305020": "Highways & Railtracks",
    "20305030": "Marine Ports & Services",
    # Consumer Discretionary
    "25101010": "Auto Parts & Equipment",
    "25101020": "Tires & Rubber",
    "25102010": "Automobile Manufacturers",
    "25102020": "Motorcycle Manufacturers",
    "25201010": "Consumer Electronics",
    "25201020": "Home Furnishings",
    "25201030": "Homebuilding",
    "25201040": "Household Appliances",
    "25201050": "Housewares & Specialties",
    "25202010": "Leisure Products",
    "25203010": "Apparel, Accessories & Luxury Goods",
    "25203020": "Footwear",
    "25203030": "Textiles",
    "25301010": "Casinos & Gaming",
    "25301020": "Hotels, Resorts & Cruise Lines",
    "25301030": "Leisure Facilities",
    "25301040": "Restaurants",
    "25302010": "Education Services",
    "25302020": "Specialized Consumer Services",
    "25501010": "Distributors",
    "25502010": "Broadline Retail",
    "25502020": "Specialty Stores",
    "25503010": "Automotive Retail",
    "25503020": "Computer & Electronics Retail",
    "25503030": "Home Improvement Retail",
    "25504010": "Homefurnishing Retail",
    "25504020": "Apparel Retail",
    "25504030": "Other Specialty Retail",
    "25504040": "Internet & Direct Marketing Retail",
    # Consumer Staples
    "30101010": "Food Distributors",
    "30101020": "Food Retail",
    "30101030": "Consumer Staples Merchandise Retail",
    "30201010": "Brewers",
    "30201020": "Distillers & Vintners",
    "30201030": "Soft Drinks & Non-alcoholic Beverages",
    "30202010": "Agricultural Products & Services",
    "30202030": "Packaged Foods & Meats",
    "30203010": "Tobacco",
    "30301010": "Household Products",
    "30302010": "Personal Care Products",
    # Health Care
    "35101010": "Health Care Equipment",
    "35101020": "Health Care Supplies",
    "35102010": "Health Care Distributors",
    "35102015": "Health Care Services",
    "35102020": "Health Care Facilities",
    "35102030": "Managed Health Care",
    "35103010": "Health Care Technology",
    "35201010": "Biotechnology",
    "35202010": "Pharmaceuticals",
    "35203010": "Life Sciences Tools & Services",
    # Financials
    "40101010": "Diversified Banks",
    "40101015": "Regional Banks",
    "40102010": "Thrifts & Mortgage Finance",
    "40201020": "Other Diversified Financial Services",
    "40201030": "Multi-Sector Holdings",
    "40201040": "Specialized Finance",
    "40202010": "Consumer Finance",
    "40203010": "Asset Management & Custody Banks",
    "40203020": "Investment Banking & Brokerage",
    "40203030": "Diversified Capital Markets",
    "40203040": "Financial Exchanges & Data",
    "40204010": "Mortgage REITs",
    "40301010": "Insurance Brokers",
    "40301020": "Life & Health Insurance",
    "40301030": "Multi-line Insurance",
    "40301040": "Property & Casualty Insurance",
    "40301050": "Reinsurance",
    # Information Technology
    "45101010": "Internet Services & Infrastructure",
    "45102010": "IT Consulting & Other Services",
    "45102020": "Internet Services & Infrastructure",
    "45102030": "Application Software",
    "45103010": "Application Software",
    "45103020": "Systems Software",
    "45103030": "Home Entertainment Software",
    "45201020": "Communications Equipment",
    "45202010": "Technology Hardware, Storage & Peripherals",
    "45202030": "Electronic Equipment & Instruments",
    "45203010": "Electronic Components",
    "45203015": "Electronic Manufacturing Services",
    "45203020": "Technology Distributors",
    "45301010": "Semiconductor Materials & Equipment",
    "45301020": "Semiconductors",
    # Communication Services
    "50101010": "Alternative Carriers",
    "50101020": "Integrated Telecommunication Services",
    "50102010": "Wireless Telecommunication Services",
    "50201010": "Advertising",
    "50201020": "Broadcasting",
    "50201030": "Cable & Satellite",
    "50201040": "Publishing",
    "50202010": "Movies & Entertainment",
    "50202020": "Interactive Home Entertainment",
    "50203010": "Interactive Media & Services",
    # Utilities
    "55101010": "Electric Utilities",
    "55102010": "Gas Utilities",
    "55103010": "Multi-Utilities",
    "55104010": "Water Utilities",
    "55105010": "Independent Power Producers & Energy Traders",
    "55105020": "Renewable Electricity",
    # Real Estate
    "60101010": "Diversified REITs",
    "60101020": "Industrial REITs",
    "60101030": "Hotel & Resort REITs",
    "60101040": "Office REITs",
    "60101050": "Health Care REITs",
    "60101060": "Residential REITs",
    "60101070": "Retail REITs",
    "60101080": "Specialized REITs",
    "60102010": "Diversified Real Estate Activities",
    "60102020": "Real Estate Operating Companies",
    "60102030": "Real Estate Development",
    "60102040": "Real Estate Services",
}


def raw_sql_with_retry(db, sql, max_retries=2):
    """Execute WRDS raw_sql with retry logic."""
    from sqlalchemy.exc import OperationalError, PendingRollbackError

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return db.raw_sql(sql)
        except PendingRollbackError as e:
            last_error = e
            conn = getattr(db, "connection", None)
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(1)
            else:
                raise
        except OperationalError as e:
            last_error = e
            if "closed the connection" in str(e):
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise
            else:
                raise
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed after {max_retries + 1} attempts")


def fetch_sec_cik_mapping():
    """Fetch SEC's official CIK-Ticker mapping as fallback for WRDS NULLs."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "name@example.com")}

    print("Fetching SEC CIK-Ticker mapping...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).replace(".", "").replace("-", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker and cik != "0000000000":
            records.append({"ticker": ticker, "cik": cik})

    df = pl.DataFrame(records)
    print(f"  Loaded {len(df)} CIK mappings from SEC")
    return df


def cik_cusip_mapping(db):
    """
    Fetch all historical CIK-CUSIP mappings from WRDS CRSP.

    Returns polars DataFrame: (permno, symbol, company, cik, cusip, start_date, end_date)
    """
    query = """
    SELECT DISTINCT
        a.kypermno,
        a.ticker,
        a.tsymbol,
        a.comnam,
        a.ncusip,
        b.cik,
        b.cikdate1,
        b.cikdate2,
        a.namedt,
        a.nameenddt
    FROM
        crsp.s6z_nam AS a
    LEFT JOIN
        wrdssec_common.wciklink_cusip AS b
        ON SUBSTR(a.ncusip, 1, 8) = SUBSTR(b.cusip, 1, 8)
        AND (b.cik IS NULL OR a.namedt <= b.cikdate2)
        AND (b.cik IS NULL OR a.nameenddt >= b.cikdate1)
    WHERE
        a.shrcd IN (10, 11)
    ORDER BY
        a.kypermno, a.namedt
    """

    print("Fetching CIK-CUSIP mapping from WRDS...")
    map_df = raw_sql_with_retry(db, query)
    map_df["namedt"] = pd.to_datetime(map_df["namedt"])
    map_df["nameenddt"] = pd.to_datetime(map_df["nameenddt"])
    map_df["cikdate1"] = pd.to_datetime(map_df["cikdate1"])
    map_df["cikdate2"] = pd.to_datetime(map_df["cikdate2"])

    # CIK validity period (longer = more reliable)
    map_df["cik_validity_days"] = (
        (map_df["cikdate2"] - map_df["cikdate1"]).apply(
            lambda x: x.total_seconds() / 86400 if pd.notnull(x) else -1
        )
    )

    # Keep most reliable CIK per (permno, symbol, namedt, nameenddt)
    map_df = map_df.sort_values(
        ["kypermno", "tsymbol", "namedt", "nameenddt", "cik_validity_days"],
        ascending=[True, True, True, True, False],
    )
    map_df = map_df.drop_duplicates(
        subset=["kypermno", "tsymbol", "namedt", "nameenddt"], keep="first"
    )

    # Group to get period ranges
    result = (
        map_df.groupby(
            ["kypermno", "cik", "ticker", "tsymbol", "comnam", "ncusip"],
            dropna=False,
        )
        .agg({"namedt": "min", "nameenddt": "max"})
        .reset_index()
        .sort_values(["kypermno", "namedt"])
        .dropna(subset=["tsymbol"])
    )

    pl_map = (
        pl.DataFrame(result)
        .with_columns(
            pl.col("kypermno").cast(pl.Int32).alias("permno"),
            pl.col("tsymbol").alias("symbol"),
            pl.col("comnam").alias("company"),
            pl.col("ncusip").alias("cusip"),
            pl.col("namedt").cast(pl.Date).alias("start_date"),
            pl.col("nameenddt").cast(pl.Date).alias("end_date"),
        )
        .select(["permno", "symbol", "company", "cik", "cusip", "start_date", "end_date"])
    )

    # SEC CIK fallback for NULLs
    null_count = pl_map.filter(pl.col("cik").is_null()).height
    total = pl_map.height
    print(f"  {null_count}/{total} records with NULL CIK from WRDS")

    if null_count > 0:
        sec_mapping = fetch_sec_cik_mapping()
        if not sec_mapping.is_empty():
            pl_map = (
                pl_map.join(sec_mapping, left_on="symbol", right_on="ticker", how="left", suffix="_sec")
                .with_columns(
                    pl.when(pl.col("cik").is_not_null())
                    .then(pl.col("cik"))
                    .otherwise(pl.col("cik_sec"))
                    .alias("cik")
                )
                .drop("cik_sec")
            )
            null_after = pl_map.filter(pl.col("cik").is_null()).height
            filled = null_count - null_after
            print(f"  SEC fallback filled {filled}/{null_count} NULL CIKs")
            print(f"  Final: {null_after}/{total} still NULL (non-SEC filers)")

    return pl_map


def security_map(cik_cusip_df):
    """
    Map security_id based on BUSINESS continuity using PERMNO.

    Rules:
    1. PERMNO changes → new security_id
    2. Same PERMNO, symbol changes, BOTH sides have non-null CIKs with no overlap → new security_id
    3. Otherwise (including null CIK on either side) → same security_id
    """
    # Group by (permno, symbol) to collect all CIKs
    period_groups = (
        cik_cusip_df.group_by(["permno", "symbol"])
        .agg([
            pl.col("cik").unique().alias("ciks"),
            pl.col("start_date").min().alias("start_date"),
            pl.col("end_date").max().alias("end_date"),
            pl.col("company").first().alias("company"),
            pl.col("cusip").first().alias("cusip"),
        ])
        .sort(["permno", "start_date"])
    )

    # Track previous period
    period_groups = period_groups.with_columns([
        pl.col("permno").shift(1).alias("prev_permno"),
        pl.col("symbol").shift(1).alias("prev_symbol"),
        pl.col("ciks").shift(1).alias("prev_ciks"),
    ])

    # Convert to pandas for CIK overlap logic
    pdf = period_groups.to_pandas()

    def cik_confirms_split(row):
        """Return True only when both sides have non-null CIKs and they don't overlap.
        Null CIK = unknown → assume same business (don't split)."""
        prev, curr = row["prev_ciks"], row["ciks"]
        if prev is None or curr is None:
            return False
        # Filter out None/null within the lists
        prev_set = {c for c in prev if c is not None}
        curr_set = {c for c in curr if c is not None}
        if not prev_set or not curr_set:
            return False
        return len(prev_set & curr_set) == 0

    pdf["confirmed_split"] = pdf.apply(cik_confirms_split, axis=1)
    pdf["new_business"] = (
        pdf["prev_permno"].isna()
        | (pdf["permno"] != pdf["prev_permno"])
        | (
            (pdf["permno"] == pdf["prev_permno"])
            & (pdf["symbol"] != pdf["prev_symbol"])
            & pdf["confirmed_split"]
        )
    )
    pdf["security_id"] = pdf["new_business"].cumsum() + 1000

    # Join security_id back to original data
    security_assignments = pl.from_pandas(pdf[["permno", "symbol", "security_id"]])
    result = (
        cik_cusip_df.join(security_assignments, on=["permno", "symbol"], how="left")
        .select([
            "security_id", "permno", "symbol", "company", "cik", "cusip",
            "start_date", "end_date",
        ])
        .with_columns(pl.col("security_id").cast(pl.Int64))
    )

    n_securities = result["security_id"].n_unique()
    n_permnos = result["permno"].n_unique()
    print(f"  Created {n_securities} security_ids from {n_permnos} PERMNOs")

    return result


def fetch_compustat_gics(db):
    """
    Fetch exchange and GICS classification from Compustat Security Daily.

    Returns polars DataFrame: (cusip6, exchange, sector, industry, subindustry)
    """
    query = """
    SELECT DISTINCT
        s.gvkey,
        SUBSTR(s.cusip, 1, 6) AS cusip6,
        s.exchg,
        c.gsector,
        c.ggroup,
        c.gsubind
    FROM comp.security s
    JOIN comp.company c ON s.gvkey = c.gvkey
    WHERE s.cusip IS NOT NULL
    """

    print("Fetching Compustat exchange + GICS data...")
    comp_df = raw_sql_with_retry(db, query)

    if comp_df.empty:
        print("  WARNING: No Compustat data returned")
        return pl.DataFrame({
            "cusip6": [], "exchange": [], "sector": [],
            "industry": [], "subindustry": [],
        })

    # Map codes to names
    comp_df["exchange"] = comp_df["exchg"].map(EXCHG_MAP)
    comp_df["sector"] = comp_df["gsector"].astype(str).map(GICS_SECTOR)
    comp_df["industry"] = comp_df["ggroup"].astype(str).map(GICS_INDUSTRY_GROUP)
    comp_df["subindustry"] = comp_df["gsubind"].astype(str).map(GICS_SUBINDUSTRY)

    result = (
        pl.DataFrame(comp_df[["cusip6", "exchange", "sector", "industry", "subindustry"]])
        .unique(subset=["cusip6"], keep="first")
    )

    print(f"  Loaded {len(result)} Compustat GICS records")
    return result


def main():
    try:
        import wrds
    except ImportError:
        print("ERROR: wrds package required. Install with: uv add wrds")
        sys.exit(1)

    output_path = Path(__file__).resolve().parent.parent / "src" / "alphalab" / "data" / "security_master.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to WRDS
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")
    if not username or not password:
        print("ERROR: Set WRDS_USERNAME and WRDS_PASSWORD environment variables")
        sys.exit(1)

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=username, wrds_password=password)

    try:
        # Step 1: CIK-CUSIP mapping
        cik_cusip_df = cik_cusip_mapping(db)

        # Step 2: Security map (assign security_ids)
        master_df = security_map(cik_cusip_df)

        # Step 3: Compustat exchange + GICS
        comp_df = fetch_compustat_gics(db)

        # Step 4: Join on cusip (first 6 chars)
        master_df = master_df.with_columns(
            pl.col("cusip").str.slice(0, 6).alias("cusip6")
        )
        master_df = (
            master_df.join(comp_df, on="cusip6", how="left")
            .drop("cusip6")
        )

        # Ensure columns exist even if Compustat had no data
        for col in ("exchange", "sector", "industry", "subindustry"):
            if col not in master_df.columns:
                master_df = master_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

        # Final column order (cusip used internally for Compustat join but dropped from output)
        master_df = master_df.select([
            "security_id", "permno", "symbol", "company", "cik",
            "start_date", "end_date", "exchange", "sector", "industry", "subindustry",
        ])

        print(f"\nSecurityMaster: {len(master_df)} rows, columns: {master_df.columns}")

        # Save as parquet with metadata
        table = master_df.to_arrow()
        metadata = {
            b"version": b"3.0",
            b"row_count": str(len(master_df)).encode(),
            b"export_timestamp": dt.datetime.utcnow().isoformat().encode(),
            b"source": b"wrds_crsp+compustat",
            b"crsp_end_date": b"2024-12-31",
        }
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing_meta, **metadata})

        pq.write_table(table, str(output_path))

        # Verify
        verify = pq.read_table(str(output_path))
        print(f"Written: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        print(f"Columns: {verify.column_names}")
        print(f"Rows: {verify.num_rows}")

        # Stats
        exchange_counts = master_df.group_by("exchange").len().sort("len", descending=True)
        print(f"\nExchange distribution:\n{exchange_counts}")

        sector_counts = master_df.filter(pl.col("sector").is_not_null()).group_by("sector").len().sort("len", descending=True)
        print(f"\nSector distribution:\n{sector_counts}")

        print("\nDone. Commit data/meta/master/security_master.parquet to the repo.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
