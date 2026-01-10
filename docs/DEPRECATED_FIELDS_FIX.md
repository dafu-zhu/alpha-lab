# Fix for Deprecated XBRL Fields

## Problem

AAPL and other companies have switched from deprecated XBRL tags to newer ones over time. For example:
- **Revenue**: `SalesRevenueNet` (deprecated 2018) → `Revenues` (current)
- **Operating Income**: `OperatingIncomeLoss` (standard)

When computing derived metrics like `op_mgn` (operating margin = operating income / revenue), data was missing after 2018 because:

1. The old `extract_concept()` function returned only the FIRST matching XBRL tag
2. For AAPL revenue, it found `SalesRevenueNet` (which stopped being reported in 2018)
3. It never looked for `Revenues` (which started being reported in 2019)
4. This created a data gap: no revenue data after 2018 → no operating margin after 2018

### Example: AAPL Revenue Timeline

```
2013-2018: AAPL reported using "SalesRevenueNet"
2019+:     AAPL reported using "Revenues"

Old behavior:
  extract_concept('rev') → Returns ONLY SalesRevenueNet data → Stops at 2018

New behavior:
  extract_concept('rev') → Merges BOTH SalesRevenueNet AND Revenues → Complete timeline
```

## Solution

Modified `extract_concept()` in `src/quantdl/collection/fundamental.py` to:

1. **Search ALL candidate tags** (not just the first match)
2. **Merge data from all matching tags** into a single dataset
3. **Deduplicate** by `(accn, frame, filed)` to avoid double-counting

### Code Changes

**Before:**
```python
for tag in tags:
    # ... validate tag ...
    if local in facts[prefix]:
        return facts[prefix][local]  # Returns FIRST match only
```

**After:**
```python
# Collect all matching fields' data
all_field_data = []
for tag in tags:
    # ... validate tag ...
    if local in facts[prefix]:
        all_field_data.append(facts[prefix][local])  # Collect ALL matches

# Merge units from all matching fields
merged = {'label': '...', 'description': '...', 'units': {}}
for field_data in all_field_data:
    for unit_type, unit_data in field_data['units'].items():
        merged['units'][unit_type].extend(unit_data)

# Deduplicate by (accn, frame, filed)
for unit_type in merged['units']:
    seen = {}
    for dp in merged['units'][unit_type]:
        key = (dp.get('accn'), dp.get('frame'), dp.get('filed'))
        if key not in seen:
            seen[key] = dp
    merged['units'][unit_type] = list(seen.values())
```

## Impact

### ✅ Fixed Issues
- Complete revenue data for companies that switched from deprecated tags
- Operating margin (`op_mgn`) and other derived metrics now have continuous data
- Works with any concept that has multiple tag mappings

### 📊 Affected Concepts

From `configs/approved_mapping.yaml`, concepts with multiple tags that benefit from this fix:

- **rev** (revenue): `SalesRevenueNet` → `Revenues`
- **cor** (cost of revenue): Multiple alternatives
- **op_inc** (operating income): Multiple alternatives
- **net_inc** (net income): Multiple alternatives
- All other concepts with multiple XBRL tag mappings

### 🔄 Backward Compatibility

- ✅ Fully backward compatible
- If only one tag matches → returns data directly (same as before)
- If multiple tags match → merges them (new behavior, fixes the gap)
- No changes to downstream code required

## Testing

To verify the fix works for AAPL:

```python
from quantdl.collection.fundamental import Fundamental

# AAPL CIK: 320193
fund = Fundamental('320193', symbol='AAPL')

# Get revenue data
rev_dps = fund.get_concept_data('rev')
rev_tuples = fund.get_value_tuple(rev_dps)

# Check date range
dates = [dt for dt, val in rev_tuples]
print(f"Revenue data range: {min(dates)} to {max(dates)}")
# Expected: Data from 2010s through 2024 (no gap at 2018)

# Get operating income
op_inc_dps = fund.get_concept_data('op_inc')

# Calculate operating margin
fields_dict = {
    'rev': fund.get_value_tuple(rev_dps),
    'op_inc': fund.get_value_tuple(op_inc_dps)
}

df = fund.collect_fields_raw('2015-01-01', '2024-12-31', fields_dict)
df = df.with_columns((df['op_inc'] / df['rev']).alias('op_mgn'))

print(df.select(['timestamp', 'op_mgn']))
# Expected: Continuous operating margin data across 2018 transition
```

## Related Files

- `src/quantdl/collection/fundamental.py` - Contains the fix
- `configs/approved_mapping.yaml` - Defines concept-to-tag mappings
- `src/quantdl/collection/models.py` - FndDataPoint data model

## References

- SEC EDGAR Taxonomy: https://www.sec.gov/info/edgar/edgartaxonomies
- US-GAAP Deprecated Elements: https://www.fasb.org/xbrl
- GitHub Issue: Derived metrics missing data points after 2018
