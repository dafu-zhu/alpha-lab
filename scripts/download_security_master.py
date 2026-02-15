"""
One-time script to download SecurityMaster from WRDS and strip CRSP-specific columns.

Run BEFORE removing wrds dependency:
    uv run python scripts/download_security_master.py

Output: data/master/security_master.parquet
Schema: security_id, symbol, company, cik, start_date, end_date
"""

import sys
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq


def main():
    try:
        import wrds
    except ImportError:
        print("ERROR: wrds package required. Install with: uv add wrds")
        sys.exit(1)

    from quantdl.master.security_master import SecurityMaster

    output_dir = Path("data/master")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "security_master.parquet"

    print("Connecting to WRDS and building SecurityMaster...")
    sm = SecurityMaster(force_rebuild=True)

    # Select only the columns we want (drop permno, cusip, exchcd)
    keep_cols = ['security_id', 'symbol', 'company', 'cik', 'start_date', 'end_date']
    available = [c for c in keep_cols if c in sm.master_tb.columns]
    master_df = sm.master_tb.select(available)

    print(f"SecurityMaster: {len(master_df)} rows, columns: {master_df.columns}")

    # Convert to Arrow and embed metadata
    table = master_df.to_arrow()
    metadata = {
        b'version': b'2.0',
        b'row_count': str(len(master_df)).encode(),
        b'export_timestamp': dt.datetime.utcnow().isoformat().encode(),
        b'source': b'wrds_crsp',
        b'schema': b'security_id,symbol,company,cik,start_date,end_date',
    }
    existing_meta = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_meta, **metadata})

    pq.write_table(table, str(output_path))

    # Verify
    verify = pq.read_table(str(output_path))
    print(f"Written: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"Columns: {verify.column_names}")
    print(f"Rows: {verify.num_rows}")

    # Sanity check: no CRSP columns leaked
    forbidden = {'permno', 'cusip', 'exchcd'}
    leaked = forbidden & set(verify.column_names)
    if leaked:
        print(f"WARNING: CRSP columns found in output: {leaked}")
        sys.exit(1)

    print("Done. Commit data/master/security_master.parquet to the repo.")
    sm.close()


if __name__ == "__main__":
    main()
