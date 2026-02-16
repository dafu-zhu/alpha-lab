# Data Fields

66 pre-built data fields stored as wide matrices (Date x security_id) in Arrow IPC format at `data/features/{field}.arrow`. Use any field name directly in alpha expressions.

Source: [`src/alphalab/features/registry.py`](../src/alphalab/features/registry.py)

---

## Price / Volume (10)

### Direct (6)

Sourced from raw daily ticks (Alpaca API, 2017+). One value per trading day per security.

| Field | Ticks Column | Description |
|-------|-------------|-------------|
| `close` | close | Adjusted close price |
| `open` | open | Adjusted open price |
| `high` | high | Intraday high |
| `low` | low | Intraday low |
| `volume` | volume | Daily share volume |
| `vwap` | vwap | Volume-weighted average price |

### Computed (4)

Derived from direct price/volume fields at build time.

| Field | Formula | Dependencies | Description |
|-------|---------|-------------|-------------|
| `returns` | `close / close.shift(1) - 1` | close | Daily simple returns |
| `adv20` | `volume.rolling_mean(20)` | volume | 20-day average daily volume |
| `cap` | `close * sharesout` | close, sharesout | Market capitalization |
| `split` | `1.0` (stub) | -- | Split adjustment factor (placeholder) |

---

## Fundamental (38)

Raw concepts from SEC EDGAR XBRL filings (2009+). Forward-filled across trading days after point-in-time alignment.

| Field | XBRL Concepts | Description |
|-------|--------------|-------------|
| `sharesout` | EntityCommonStockSharesOutstanding, CommonStockSharesOutstanding, SharesOutstanding | Shares outstanding |
| `dividend` | PaymentsOfDividendsCommonStock, PaymentsOfDividends, DividendsCommonStockCash | Dividends paid |
| `assets` | Assets | Total assets |
| `liabilities` | Liabilities | Total liabilities |
| `operating_income` | OperatingIncomeLoss, ProfitLossFromOperatingActivities | Operating income |
| `sales` | SalesRevenueNet, Revenues, Revenue, RevenueFromContractWithCustomerExcludingAssessedTax | Total revenue |
| `capex` | CapitalExpendituresIncurredButNotYetPaid, PaymentsToAcquirePropertyPlantAndEquipment | Capital expenditures |
| `equity` | StockholdersEquity, Equity | Total stockholders' equity |
| `debt_lt` | LongTermDebt, LongTermDebtNoncurrent | Long-term debt |
| `assets_curr` | AssetsCurrent, PrepaidExpenseAndOtherAssetsCurrent | Current assets |
| `goodwill` | Goodwill | Goodwill |
| `income` | NetIncomeLoss, ProfitLoss, NetIncomeLossAvailableToCommonStockholdersBasic | Net income |
| `revenue` | SalesRevenueNet, Revenues, Revenue, RevenueFromContractWithCustomerExcludingAssessedTax | Total revenue (alias of sales) |
| `cashflow_op` | NetCashProvidedByUsedInOperatingActivities, CashFlowsFromUsedInOperatingActivities | Operating cash flow |
| `cash` | CashAndCashEquivalentsAtCarryingValue, CashAndCashEquivalents | Cash and equivalents |
| `cogs` | CostOfRevenue, CostOfGoodsSold, CostOfSales | Cost of goods sold |
| `liabilities_curr` | LiabilitiesCurrent, OtherLiabilitiesCurrent | Current liabilities |
| `debt_st` | ShortTermBorrowings | Short-term debt |
| `ppent` | PropertyPlantAndEquipmentNet, PropertyPlantAndEquipmentGross | PP&E net |
| `cashflow` | NetCashProvidedByUsedInOperatingActivities, CashFlowsFromUsedInOperatingActivities | Operating cash flow (alias of cashflow_op) |
| `inventory` | InventoryNet, Inventories | Inventory |
| `cash_st` | CashAndCashEquivalentsAtCarryingValue, CashAndCashEquivalents | Cash and equivalents (alias of cash) |
| `receivable` | AccountsReceivableNetCurrent, NotesReceivableNet, TradeAndOtherCurrentReceivables | Accounts receivable |
| `sga_expense` | SellingGeneralAndAdministrativeExpense | SG&A expense |
| `retained_earnings` | RetainedEarningsAccumulatedDeficit | Retained earnings |
| `cashflow_fin` | NetCashProvidedByUsedInFinancingActivities, CashFlowsFromUsedInFinancingActivities | Financing cash flow |
| `income_tax` | IncomeTaxExpenseBenefit, DeferredIncomeTaxExpenseBenefit | Income tax expense |
| `pretax_income` | IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest, ProfitLossBeforeTax | Pretax income |
| `cashflow_invest` | NetCashProvidedByUsedInInvestingActivities, CashFlowsFromUsedInInvestingActivities | Investing cash flow |
| `cashflow_dividends` | PaymentsOfDividendsCommonStock, PaymentsOfDividends, DividendsCommonStockCash | Dividends paid (alias of dividend) |
| `depre_amort` | DepreciationDepletionAndAmortization, Depreciation, DepreciationAndAmortization | Depreciation and amortization |

### Extended Concepts (7)

Additional SEC EDGAR concepts mapped via `sec_mapping.yaml`.

| Field | Registry Name | XBRL Concepts | Description |
|-------|-------------|--------------|-------------|
| `fnd6_drft` | dr_lt | DeferredRevenueNoncurrent, ContractWithCustomerLiabilityNoncurrent | Deferred revenue (long-term) |
| `fnd6_drc` | dr_st | DeferredRevenueCurrent, ContractWithCustomerLiabilityCurrent | Deferred revenue (current) |
| `fnd6_ivaco` | invest_activity_other | PaymentsForProceedsFromOtherInvestingActivities, OtherPaymentsToAcquireBusinesses | Investing activities (other) |
| `income_beforeextra` | income_before_extra | IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest, IncomeLossFromContinuingOperations | Income before extraordinary items |
| `fnd6_acdo` | assets_discontinued | AssetsOfDisposalGroupIncludingDiscontinuedOperationCurrent | Discontinued operations assets |
| `fnd6_clother` | other_income_nt | OtherComprehensiveIncomeLossNetOfTax | Other comprehensive income (net of tax) |
| `fnd6_xrent` | rental_expense | OperatingLeaseExpense, LeaseAndRentalExpense | Rental/lease expense |

---

## Derived (14)

Arithmetic combinations of fundamental and price fields. Built after dependencies via topological sort.

| Field | Formula | Dependencies | Description |
|-------|---------|-------------|-------------|
| `debt` | `debt_lt + debt_st` | debt_lt, debt_st | Total debt |
| `invested_capital` | `equity + debt_lt - cash` | equity, debt_lt, cash | Invested capital |
| `enterprise_value` | `cap + debt_lt - cash` | cap, debt_lt, cash | Enterprise value |
| `ebitda` | `operating_income + depre_amort` | operating_income, depre_amort | EBITDA |
| `eps` | `income / sharesout` | income, sharesout | Earnings per share |
| `bookvalue_ps` | `equity / sharesout` | equity, sharesout | Book value per share |
| `operating_expense` | `cogs + sga_expense + depre_amort` | cogs, sga_expense, depre_amort | Total operating expense |
| `current_ratio` | `assets_curr / liabilities_curr` | assets_curr, liabilities_curr | Current ratio |
| `return_equity` | `income / equity` | income, equity | Return on equity |
| `return_assets` | `income / assets` | income, assets | Return on assets |
| `sales_ps` | `sales / sharesout` | sales, sharesout | Sales per share |
| `inventory_turnover` | `cogs / inventory` | cogs, inventory | Inventory turnover |
| `working_capital` | `assets_curr - liabilities_curr` | assets_curr, liabilities_curr | Working capital |
| `sales_growth` | `sales / sales.shift(63) - 1` | sales | Quarterly sales growth (~63 trading days) |

Division operations return null when the denominator is zero.

---

## Group (4)

Categorical labels from SecurityMaster. Constant per security across all trading days (Utf8 columns).

| Field | Source Column | Description |
|-------|-------------|-------------|
| `sector` | sector | GICS sector classification |
| `industry` | industry | GICS industry classification |
| `subindustry` | subindustry | GICS sub-industry classification |
| `exchange` | exchange | Listing exchange (NYSE, NASDAQ, etc.) |

GICS classifications sourced from yfinance, mapped via `configs/morningstar_to_gics.yaml`.

---

## Field Counts Summary

| Category | Count |
|----------|-------|
| Price / Volume (direct) | 6 |
| Price / Volume (computed) | 4 |
| Fundamental (raw) | 31 |
| Fundamental (extended) | 7 |
| Derived | 14 |
| Group | 4 |
| **Total** | **66** |

---

## Output Format

All fields are written as Arrow IPC files at `$LOCAL_STORAGE_PATH/data/features/{field}.arrow`.

- **Schema:** `Date` (date) + one column per `security_id` (Float64 for numeric, Utf8 for group)
- **Alignment:** All fields share the same trading day calendar (rows) and security_id set (columns)
- **Fundamental fill:** Point-in-time values forward-filled across trading days
- **Build order:** Topological sort ensures dependencies are built before derived fields

```python
from alphalab.features.registry import ALL_FIELDS, get_build_order

# List all field names
print(sorted(ALL_FIELDS.keys()))

# Get dependency-safe build order
order = get_build_order()
```
