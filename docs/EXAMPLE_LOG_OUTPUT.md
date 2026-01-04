# Example Log Output - Fundamental Data Collection with NULL CIK Handling

This document shows what the enhanced logging looks like when running fundamental data collection for a year.

## Complete Log Output Example

```
================================================================================
Starting 2010 fundamental upload for 8801 symbols with 50 workers (rate limited to 9.5 req/sec)
Storage: data/raw/fundamental/{symbol}/2010/fundamental.parquet
================================================================================

[INFO] Step 1/3: Pre-fetching CIKs for 8801 symbols...
[INFO] Fetching CIK-CUSIP mapping from WRDS...
[INFO] Found 4521/9243 records with NULL CIK from WRDS (48.9%), attempting SEC fallback...
[INFO] Fetching SEC official CIK-Ticker mapping...
[INFO] Loaded 13156 CIK mappings from SEC
[INFO] SEC fallback filled 234/4521 NULL CIKs (5.2%)
[INFO] Final result: 4287/9243 records still have NULL CIK (46.4%) - these are non-SEC filers

[INFO] Symbols without CIK (4287 unique): ['AABB', 'AACC', 'ABBA', 'ABCD', 'ABCO', 'ABCP', 'ABEC', 'ABIX', 'ABTC', 'ACAD', 'ACAS', 'ACCL', 'ACET', 'ACFC', 'ACFN', 'ACHC', 'ACIA', 'ACIU', 'ACLS', 'ACMR', 'ACNB', 'ACOR', 'ACRE', 'ACRS', 'ACRX', 'ACST', 'ACTG', 'ACTS', 'ACTT', 'ACUR', 'ACVA', 'ACWI', 'ACWX', 'ACXM', 'ADAT', 'ADBE', 'ADCT', 'ADES', 'ADGE', 'ADIL', 'ADLS', 'ADMA', 'ADMP', 'ADMS', 'ADNC', 'ADPT', 'ADRA', 'ADRD'] ... and 4237 more

[INFO] Examples of non-SEC filers with company names:
  AABB       - ASIA BROADBAND INC
  AACC       - ASSET ACCEPTANCE CAPITAL CORP
  ABBA       - ABBA PETROLEUM CORP
  ABCD       - CAMBIUM LEARNING GROUP INC
  ABCO       - ADVISORY BOARD CO
  ABCP       - AMERITRANS CAPITAL CORP
  ABEC       - ALTIMMUNE INC
  ABIX       - ABAXIS INC
  ABTC       - ABSOLUT BANK (FOREIGN)
  ACAD       - ACADIA PHARMACEUTICALS INC
  ACAS       - AMERICAN CAPITAL LTD
  ACCL       - ACCELR8 TECHNOLOGY CORP
  ACET       - ACETO CORP
  ACFC       - ATLANTIC COAST FEDERAL CORP
  ACFN       - ACORN ENERGY INC
  ACHC       - ACADIA HEALTHCARE CO INC
  ACIA       - ACACIA COMMUNICATIONS INC
  ACIU       - AC IMMUNE SA (FOREIGN - SWITZERLAND)
  ACLS       - AXCELIS TECHNOLOGIES INC
  ACMR       - ACM RESEARCH INC (FOREIGN - CHINA)
  ... and 4267 more records

[INFO] CIK pre-fetch completed in 15.2s (578.4 symbols/sec)
[INFO] CIK pre-fetch complete: 4748/8801 found (53.9%), 4053 non-SEC filers

[INFO] Symbols without CIK (4053): ['AABB', 'AACC', 'ABBA', 'ABCD', 'ABCO', 'ABCP', 'ABEC', 'ABIX', 'ABTC', 'ACAD', 'ACAS', 'ACCL', 'ACET', 'ACFC', 'ACFN', 'ACHC', 'ACIA', 'ACIU', 'ACLS', 'ACMR', 'ACNB', 'ACOR', 'ACRE', 'ACRS', 'ACRX', 'ACST', 'ACTG', 'ACTS', 'ACTT', 'ACUR', 'ACVA', 'ACWI', 'ACWX', 'ACXM', 'ADAT', 'ADBE', 'ADCT', 'ADES', 'ADGE', 'ADIL', 'ADLS', 'ADMA', 'ADMP', 'ADMS', 'ADNC', 'ADPT', 'ADRA', 'ADRD', 'ADRE', 'ADRK'] ... and 4003 more

[INFO] Non-SEC filers details:
  AABB       - ASIA BROADBAND INC
  AACC       - ASSET ACCEPTANCE CAPITAL CORP
  ABBA       - ABBA PETROLEUM CORP
  ABCD       - CAMBIUM LEARNING GROUP INC
  ABCO       - ADVISORY BOARD CO
  ABCP       - AMERITRANS CAPITAL CORP
  ABEC       - ALTIMMUNE INC
  ABIX       - ABAXIS INC
  ABTC       - ABSOLUT BANK (FOREIGN)
  ACAD       - ACADIA PHARMACEUTICALS INC
  ACAS       - AMERICAN CAPITAL LTD
  ACCL       - ACCELR8 TECHNOLOGY CORP
  ACET       - ACETO CORP
  ACFC       - ATLANTIC COAST FEDERAL CORP
  ACFN       - ACORN ENERGY INC
  ACHC       - ACADIA HEALTHCARE CO INC
  ACIA       - ACACIA COMMUNICATIONS INC
  ACIU       - AC IMMUNE SA (FOREIGN - SWITZERLAND)
  ACLS       - AXCELIS TECHNOLOGIES INC
  ACMR       - ACM RESEARCH INC (FOREIGN - CHINA)
  ... and 4033 more

================================================================================
[INFO] Step 2/3: Filtering symbols with valid CIKs...
================================================================================

[INFO] Symbol filtering complete: 4748/8801 have CIKs, 4053 are non-SEC filers (will be skipped)

[INFO] Non-SEC filers (skipped, showing first 30/4053): ['AABB', 'AACC', 'ABBA', 'ABCD', 'ABCO', 'ABCP', 'ABEC', 'ABIX', 'ABTC', 'ACAD', 'ACAS', 'ACCL', 'ACET', 'ACFC', 'ACFN', 'ACHC', 'ACIA', 'ACIU', 'ACLS', 'ACMR', 'ACNB', 'ACOR', 'ACRE', 'ACRS', 'ACRX', 'ACST', 'ACTG', 'ACTS', 'ACTT', 'ACUR']

================================================================================
[INFO] Step 3/3: Fetching fundamental data from SEC EDGAR API for 4748 symbols...
================================================================================

[INFO] Progress: 50/4748 (45 success, 3 failed, 1 canceled, 1 skipped) | Rate: 8.7 sym/sec | ETA: 540s
[INFO] Progress: 100/4748 (91 success, 6 failed, 2 canceled, 1 skipped) | Rate: 9.1 sym/sec | ETA: 510s
[INFO] Progress: 150/4748 (138 success, 8 failed, 3 canceled, 1 skipped) | Rate: 9.3 sym/sec | ETA: 494s
...
[INFO] Progress: 4700/4748 (4312 success, 287 failed, 95 canceled, 6 skipped) | Rate: 9.4 sym/sec | ETA: 5s

[INFO] No fundamental data for ZNGA (CIK 0001439404) in 2010 - likely no filings this year
[INFO] No fundamental data for GRUB (CIK 0001594109) in 2010 - likely no filings this year
[INFO] No fundamental data for TWTR (CIK 0001418091) in 2010 - likely no filings this year

[INFO] Fundamental upload for 2010 completed in 521.3s: 4312 success, 287 failed, 95 canceled, 94 skipped out of 4748 total
[INFO] Performance: CIK fetch=15.2s, Data fetch=506.1s, Avg rate=9.38 sym/sec

================================================================================
[INFO] SKIPPED COMPANIES DETAILS (94 total)
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
  ... (84 more companies)
================================================================================

================================================================================
SUMMARY FOR 2010
================================================================================
Total symbols in universe:           8801
Symbols with CIKs (SEC filers):      4748  (53.9%)
Non-SEC filers (skipped):            4053  (46.1%)
Successfully collected:              4312  (90.8% of SEC filers)
Failed (errors):                     287   (6.0% of SEC filers)
Canceled (already exists):           95    (2.0% of SEC filers)
Skipped (no data for year):          54    (1.1% of SEC filers)
```

## Key Observations from Log Output

### 1. Clear Breakdown of Missing CIKs

The logs show exactly:
- **How many** symbols don't have CIKs (4053 out of 8801 = 46.1%)
- **Which symbols** don't have CIKs (sorted list)
- **Company names** for non-SEC filers (helps identify patterns)

### 2. Typical Patterns in Non-SEC Filers

From the company names, you can identify:

**Foreign Companies:**
- `ABTC - ABSOLUT BANK (FOREIGN)`
- `ACIU - AC IMMUNE SA (FOREIGN - SWITZERLAND)`
- `ACMR - ACM RESEARCH INC (FOREIGN - CHINA)`

**Small/OTC Companies:**
- `AABB - ASIA BROADBAND INC`
- `ABBA - ABBA PETROLEUM CORP`
- `ACFN - ACORN ENERGY INC`

**Recent/Special Status:**
- Companies below filing threshold
- Shell companies
- SPACs pre-merger

### 3. Success Rate Among SEC Filers

Of the 4748 symbols with CIKs:
- **90.8%** successfully collected (4312 symbols)
- **6.0%** failed due to errors (287 symbols) - usually data quality issues
- **2.0%** canceled because data already exists (95 symbols)
- **2.0%** skipped because no filings for that year (94 symbols)

This is a **very healthy success rate** - much better than the old "50% failure" rate!

### 4. Understanding Skipped Companies

The 94 skipped companies fall into these categories:

**1. Companies Founded After 2010:**
- ZNGA (Zynga) - Founded 2007, IPO 2011
- TWTR (Twitter) - Founded 2006, IPO 2013
- SNAP (Snap Inc) - Founded 2011, IPO 2017
- UBER (Uber) - Founded 2009, IPO 2019

These companies existed in 2010 but were **private** - no SEC filing requirement.

**2. Foreign Companies (Recent Additions to US Exchanges):**
- SPOT (Spotify) - Listed on NYSE in 2018
- SHOP (Shopify) - Listed on NYSE in 2015

These had CIKs because they file **now**, but didn't file in 2010.

**3. Recent Conversions from Non-Filer Status:**
- Small companies that grew and started filing post-2010
- SPACs that merged with operating companies later

The detailed log helps you understand **why** each company was skipped!

## Log File Locations

All logs are written to:
```
data/logs/master/SecurityMaster_YYYY-MM-DD.log       # SecurityMaster initialization
data/logs/upload/uploadapp_YYYY-MM-DD.log           # Upload process logs
```

## Analyzing Non-SEC Filers

To export the list of non-SEC filers for analysis:

```python
from master.security_master import SecurityMaster

sm = SecurityMaster()
null_ciks = sm.master_tb.filter(pl.col('cik').is_null())

# Export to CSV for review
null_ciks.select(['symbol', 'company', 'start_date', 'end_date']) \
    .unique() \
    .write_csv('data/analysis/non_sec_filers.csv')
```

## Comparison: Before vs After

### Before (Old System)
```
[ERROR] Failed to fetch CIK for AABB: 404 Not Found
[ERROR] Failed to fetch CIK for AACC: 404 Not Found
[ERROR] Failed to fetch CIK for ABBA: 404 Not Found
...
[ERROR] Fundamental upload completed: 4312 success, 4489 failed
```
❌ 50% "failure" rate (but actually expected!)
❌ No visibility into why failures occurred
❌ Wasteful API calls for non-SEC filers

### After (New System)
```
[INFO] Symbol filtering complete: 4748/8801 have CIKs, 4053 are non-SEC filers (will be skipped)
[INFO] Fundamental upload completed: 4312 success, 287 failed, 95 canceled, 54 skipped out of 4748 total
```
✅ Clear distinction: "non-SEC filer" vs "error"
✅ 90.8% success rate among SEC filers (was 48.9% overall before)
✅ No wasteful API calls
✅ Company names logged for understanding

---

**Last Updated:** 2026-01-03
