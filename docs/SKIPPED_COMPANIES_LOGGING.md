# Skipped Companies Detailed Logging

## Overview

This document explains the comprehensive logging system for tracking which companies are skipped during fundamental data collection and **why** they are skipped.

## Two Types of "Skipped"

### 1. Non-SEC Filers (Filtered Before Collection)

**Count:** ~46% of all US-listed stocks (~4,053 out of 8,801)

**Reason:** No CIK available (don't file with SEC)

**When:** Filtered in Step 2/3 before making any API calls

**Examples:**
- Foreign companies (use Form 6-K, not 10-K)
- Small companies (below $10M asset threshold)
- OTC/Pink Sheet stocks
- Non-operating entities (shells, SPACs)

**Logged at:**
- `SecurityMaster.cik_cusip_mapping()` - Shows symbols without CIKs with company names
- `CIKResolver.batch_prefetch_ciks()` - Shows non-SEC filers with company names
- `UploadApp.fundamental()` Step 2/3 - Shows filtered symbols

### 2. Companies with CIKs but No Data for Specific Year (Skipped During Collection)

**Count:** ~2% of SEC filers (~94 out of 4,748 in example)

**Reason:** Company has a CIK but didn't file in that specific year

**When:** During Step 3/3 (SEC EDGAR API calls)

**Examples:**
- **Private companies that went public later:**
  - ZNGA (Zynga) - Founded 2007, IPO 2011 → No 2010 filings
  - TWTR (Twitter) - Founded 2006, IPO 2013 → No 2010 filings
  - UBER (Uber) - Founded 2009, IPO 2019 → No 2010 filings

- **Foreign companies that listed later:**
  - SPOT (Spotify) - Listed NYSE 2018 → No 2010 filings
  - SHOP (Shopify) - Listed NYSE 2015 → No 2010 filings

- **Companies that started filing later:**
  - Small companies that grew above filing threshold
  - SPACs that merged with operating companies

**Logged at:**
- Real-time: `DataPublishers.publish_fundamental()` - Logs each skip as it happens
- Summary: `UploadApp.fundamental()` - Shows complete list with company names at end

## Logging Output Example

### During Collection

```
[INFO] No fundamental data for ZNGA (CIK 0001439404) in 2010 - likely no filings this year
[INFO] No fundamental data for GRUB (CIK 0001594109) in 2010 - likely no filings this year
[INFO] No fundamental data for TWTR (CIK 0001418091) in 2010 - likely no filings this year
```

### Summary Report

```
================================================================================
SKIPPED COMPANIES DETAILS (94 total)
================================================================================
  ZNGA       - ZYNGA INC                                        (CIK 0001439404)
             Reason: No fundamental data available for ZNGA in 2010
  GRUB       - GRUBHUB INC                                      (CIK 0001594109)
             Reason: No fundamental data available for GRUB in 2010
  TWTR       - TWITTER INC                                      (CIK 0001418091)
             Reason: No fundamental data available for TWTR in 2010
  SNAP       - SNAP INC                                         (CIK 0001564408)
             Reason: No fundamental data available for SNAP in 2010
  UBER       - UBER TECHNOLOGIES INC                            (CIK 0001543151)
             Reason: No fundamental data available for UBER in 2010
  LYFT       - LYFT INC                                         (CIK 0001759509)
             Reason: No fundamental data available for LYFT in 2010
  PINS       - PINTEREST INC                                    (CIK 0001506126)
             Reason: No fundamental data available for PINS in 2010
  DDOG       - DATADOG INC                                      (CIK 0001561550)
             Reason: No fundamental data available for DDOG in 2010
  SPOT       - SPOTIFY TECHNOLOGY SA                            (CIK 0001639920)
             Reason: No fundamental data available for SPOT in 2010
  SHOP       - SHOPIFY INC                                      (CIK 0001594805)
             Reason: No fundamental data available for SHOP in 2010
  SQ         - SQUARE INC                                       (CIK 0001512673)
             Reason: No fundamental data available for SQ in 2010
  ROKU       - ROKU INC                                         (CIK 0001428439)
             Reason: No fundamental data available for ROKU in 2010
  ZM         - ZOOM VIDEO COMMUNICATIONS INC                    (CIK 0001585521)
             Reason: No fundamental data available for ZM in 2010
  DOCU       - DOCUSIGN INC                                     (CIK 0001261333)
             Reason: No fundamental data available for DOCU in 2010
  ... (80 more companies)
================================================================================
```

## Implementation Details

### 1. Data Publishers (`src/storage/data_publishers.py`)

**Modified `publish_fundamental()` method:**

```python
# Check if DataFrame is empty
if len(combined_df) == 0:
    cik_str = f" (CIK {cik})" if cik else ""
    self.logger.info(f'No fundamental data for {sym}{cik_str} in {year} - likely no filings this year')
    return {
        'symbol': sym,
        'status': 'skipped',
        'error': f'No fundamental data available for {sym} in {year}',
        'cik': cik  # Include CIK in result for logging
    }
```

### 2. Upload App (`src/storage/upload_app.py`)

**Track skipped symbols during collection:**

```python
# Track skipped symbols with details for logging
skipped_symbols = []  # List of {symbol, cik, error} dicts

# When processing results:
elif result['status'] == 'skipped':
    skipped += 1
    skipped_symbols.append({
        'symbol': result.get('symbol'),
        'cik': result.get('cik'),
        'error': result.get('error', 'Unknown reason')
    })
```

**Generate detailed summary at end:**

```python
# Log detailed information about skipped symbols
if skipped > 0:
    self.logger.info(f"\n{'='*80}")
    self.logger.info(f"SKIPPED COMPANIES DETAILS ({skipped} total)")
    self.logger.info(f"{'='*80}")

    # Get company names from SecurityMaster
    skipped_symbol_list = [s['symbol'] for s in skipped_symbols]
    master_tb = self.crsp_ticks.security_master.master_tb
    skipped_details = master_tb.filter(
        pl.col('symbol').is_in(skipped_symbol_list)
    ).select(['symbol', 'company', 'cik']).unique()

    # Log each skipped company with details
    for skip_info in skipped_symbols:
        sym = skip_info['symbol']
        cik = skip_info['cik']
        reason = skip_info['error']

        # Fetch company name from SecurityMaster
        company_match = skipped_details.filter(pl.col('symbol') == sym)
        company = company_match['company'].head(1).item() if not company_match.is_empty() else "Unknown"

        cik_str = f"CIK {cik}" if cik else "No CIK"
        self.logger.info(f"  {sym:10} - {company:50} ({cik_str})")
        self.logger.info(f"             Reason: {reason}")
```

## Use Cases

### 1. Understanding Data Coverage Gaps

When you see "94 skipped" in your logs, you now know:
- **Which** companies were skipped (symbol + name)
- **Why** they were skipped (no filings for that year)
- **CIK** for each (confirms they are SEC filers, just not in that year)

### 2. Validating Data Collection Logic

If you see unexpected skips:
- Check if company existed in that year (IPO date)
- Verify if company was public vs private
- Confirm filing status on SEC EDGAR website

### 3. Reporting Data Quality

For documentation:
```
For 2010:
- Total symbols: 8,801
- SEC filers: 4,748 (53.9%)
- Successfully collected: 4,312 (90.8% of SEC filers)
- Skipped (no 2010 filings): 94 (2.0% of SEC filers)
  - Mostly pre-IPO companies (ZNGA, TWTR, UBER, etc.)
```

## Comparison: Before vs After

### Before (No Detailed Logging)
```
Progress: 150/4189 (37 success, 19 failed, 0 canceled, 94 skipped)
```
❌ No visibility into which companies were skipped
❌ No understanding of why they were skipped
❌ Can't distinguish expected skips from errors

### After (Detailed Logging)
```
Progress: 150/4748 (37 success, 19 failed, 0 canceled, 94 skipped)
...
SKIPPED COMPANIES DETAILS (94 total)
  ZNGA - ZYNGA INC (CIK 0001439404)
       Reason: No fundamental data available for ZNGA in 2010
  ...
```
✅ Every skipped company listed with name
✅ Clear reason for each skip
✅ CIK shown to confirm they are SEC filers
✅ Can validate skips are expected (pre-IPO companies)

## Expected Skip Patterns by Year

### 2010-2012 (Many Tech Startups Pre-IPO)
High skip rate expected:
- ZNGA, TWTR, SNAP, UBER, LYFT, PINS, DDOG, SPOT, SHOP, etc.

### 2015-2018 (Fewer Private Startups)
Lower skip rate expected:
- Most major tech companies had IPO'd
- Mainly foreign companies and small SPACs

### 2020+ (Very Low Skip Rate)
Minimal skips expected:
- Most current public companies have full filing history

## Files Modified

1. **src/storage/data_publishers.py**
   - Added CIK to skip result dict
   - Added real-time logging for skips

2. **src/storage/upload_app.py**
   - Track skipped symbols during collection
   - Generate detailed summary with company names
   - Fetch company names from SecurityMaster

3. **docs/EXAMPLE_LOG_OUTPUT.md**
   - Updated with skip details example
   - Added explanation of skip patterns

---

**Last Updated:** 2026-01-03
**Related Documents:**
- `NULL_CIK_HANDLING.md` - Explains non-SEC filers (Type 1 skips)
- `EXAMPLE_LOG_OUTPUT.md` - Complete log output example
