# NULL CIK Handling Design

## Overview

This document describes how the system handles stocks without CIKs (Central Index Keys) in the SEC EDGAR database. Approximately 40-50% of US-listed stocks do not file with the SEC and therefore have no CIK.

## Why Are CIKs Missing?

Not all US-listed stocks file with the SEC:

1. **Foreign companies** - Use Form 6-K instead of 10-K/10-Q (no fundamental data)
2. **Small companies** - Below $10M asset threshold (exempt from Form 10-K)
3. **OTC/Pink Sheet stocks** - Below SEC filing requirements
4. **Non-operating entities** - Shell companies, SPACs pre-merger
5. **Recent IPOs** - Haven't filed their first quarterly/annual report yet

## Data Flow & Handling Strategy

### 1. SecurityMaster: Two-Tier CIK Resolution

**Location:** `src/master/security_master.py:231` (`cik_cusip_mapping()`)

**Strategy:**
1. **Primary source (WRDS):** Query `wrdssec_common.wciklink_cusip` for historical CIK-CUSIP mappings
   - When WRDS has a CIK, it is **authoritative and accurate**
   - Forward-fill within same PERMNO/CUSIP to handle stale data

2. **Fallback source (SEC API):** For NULL CIKs from WRDS, query SEC's official ticker→CIK mapping
   - Fetches from `https://www.sec.gov/files/company_tickers.json`
   - Current snapshot only (not historical), so best-effort match
   - Cached to avoid repeated API calls

3. **Accept NULL:** If both sources return NULL, keep CIK as NULL
   - These are confirmed non-SEC filers
   - Do NOT attempt to collect fundamental data for these symbols

**Logging:**
```
Fetching CIK-CUSIP mapping from WRDS...
Found 4521/9243 records with NULL CIK from WRDS (48.9%), attempting SEC fallback...
Fetching SEC official CIK-Ticker mapping...
Loaded 13156 CIK mappings from SEC
SEC fallback filled 234/4521 NULL CIKs (5.2%)
Final result: 4287/9243 records still have NULL CIK (46.4%) - these are non-SEC filers

Symbols without CIK (4287 unique): ['AABB', 'AACC', 'ABCD', 'ACME', ... and 4237 more]
Examples of non-SEC filers with company names:
  AABB       - ABC BANCORP LTD (FOREIGN)
  ACME       - ACME MINING CO
  BCDE       - BATTERY CORP
  CDEF       - CHINA DEVELOPMENT FUND
  ... and 4267 more records
```

### 2. CIKResolver: Graceful NULL Handling

**Location:** `src/storage/cik_resolver.py`

**Changes:**

#### `get_cik()` method (line 37):
- Returns `None` immediately when NULL CIK found in SecurityMaster
- Logs debug message: `"Symbol X has NULL CIK - likely non-SEC filer"`
- Does NOT retry with fallback dates (NULL is expected, not an error)

#### `batch_prefetch_ciks()` method (line 127):
- Returns dictionary: `{symbol: Optional[str]}`
- Counts and logs non-SEC filers separately:
  ```
  CIK pre-fetch complete: 5123/9243 found (55.4%), 4120 non-SEC filers
  ```

### 3. UploadApp: Skip Non-SEC Filers

**Location:** `src/storage/upload_app.py:466` (`fundamental()` method)

**Strategy:**

**Step 1:** Batch pre-fetch all CIKs
```python
cik_map = self.cik_resolver.batch_prefetch_ciks(sec_symbols, year)
# Returns: {'AAPL': '0000320193', 'INVALID': None, ...}
```

**Step 2:** Filter to only symbols with valid CIKs
```python
symbols_with_cik = [sym for sym in sec_symbols if cik_map.get(sym) is not None]
symbols_without_cik = [sym for sym in sec_symbols if cik_map.get(sym) is None]
```

**Step 3:** Process only symbols with CIKs
- Only submit tasks for `symbols_with_cik`
- Non-SEC filers are completely skipped (no API calls, no errors)

**Logging:**
```
Step 1/3: Pre-fetching CIKs for 8801 symbols...
CIK pre-fetch completed in 12.3s (715.1 symbols/sec)
CIK pre-fetch complete: 4748/8801 found (53.9%), 4053 non-SEC filers

Symbols without CIK (4053): ['AABB', 'ACME', 'BCDE', ... and 4003 more]
Non-SEC filers details:
  AABB       - ABC BANCORP LTD
  ACME       - ACME MINING CO
  BCDE       - BATTERY CORP
  CDEF       - CHINA DEVELOPMENT FUND
  ... and 4033 more

Step 2/3: Filtering symbols with valid CIKs...
Symbol filtering complete: 4748/8801 have CIKs, 4053 are non-SEC filers (will be skipped)
Non-SEC filers (skipped, showing first 30/4053): ['AABB', 'ACME', 'BCDE', ...]

Step 3/3: Fetching fundamental data from SEC EDGAR API for 4748 symbols...
```

## Impact on Data Coverage

### What This Means for Users

1. **Fundamental data** will only be available for ~50-55% of US-listed stocks
   - This is **correct and expected** - you cannot get 10-K data for companies that don't file 10-Ks
   - No survivorship bias introduced (historical symbols retained, just filtered)

2. **Daily/minute tick data** will still be available for all stocks
   - Price data comes from CRSP/Alpaca, not SEC
   - NULL CIK only affects fundamental data collection

3. **Query API considerations**:
   - When querying fundamentals, expect ~50% of symbols to have no data
   - This should be clearly documented in the data dictionary
   - Consider adding a `has_fundamentals` flag to ticker metadata

## Testing Recommendations

### Unit Tests

1. Test `SecurityMaster._fetch_sec_cik_mapping()`:
   - Verify API call succeeds
   - Check caching works
   - Handle network failures gracefully

2. Test `SecurityMaster.cik_cusip_mapping()`:
   - Verify NULL preservation (don't replace NULL with empty string)
   - Check WRDS CIKs take precedence over SEC fallback
   - Verify logging is clear

3. Test `CIKResolver.get_cik()`:
   - Return `None` for NULL CIKs (not exception)
   - Verify no unnecessary fallback date attempts

4. Test `UploadApp.fundamental()`:
   - Verify symbols without CIKs are skipped
   - Check final statistics exclude non-SEC filers
   - No failed API calls for NULL CIK symbols

### Integration Tests

1. Run on a small sample with known non-SEC filers
2. Verify no 404 errors from SEC API for NULL CIK symbols
3. Check logs clearly distinguish "not found" vs "non-SEC filer"

## Future Enhancements

1. **Ticker metadata enrichment**:
   - Add `has_cik: bool` column to `ticker_metadata.parquet`
   - Add `sec_filer: bool` flag for query filtering

2. **Documentation**:
   - Update data dictionary to explain NULL fundamental coverage
   - Add FAQ: "Why does symbol X have no fundamental data?"

3. **Query API**:
   - Provide method to list all SEC filers: `api.get_sec_filers(year=2024)`
   - Add filter parameter: `api.query_fundamentals(..., sec_filers_only=True)`

## Rollback Plan

If SEC API fallback causes issues:
1. Remove SEC fallback logic from `cik_cusip_mapping()` (lines 307-349)
2. Keep NULL handling in `CIKResolver` and `UploadApp` (still needed for WRDS NULLs)
3. Accept slightly higher NULL rate (~48% instead of ~46%)

---

**Last Updated:** 2026-01-03
**Related Files:**
- `src/master/security_master.py`
- `src/storage/cik_resolver.py`
- `src/storage/upload_app.py`
