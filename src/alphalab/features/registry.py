"""Central field registry mapping WQ field names to data sources and build metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from graphlib import TopologicalSorter


class FieldCategory(Enum):
    PRICE_VOLUME = auto()
    FUNDAMENTAL = auto()
    DERIVED = auto()
    GROUP = auto()


class FieldSource(Enum):
    TICKS = auto()
    FUNDAMENTAL_RAW = auto()
    COMPUTED = auto()
    METADATA = auto()


@dataclass(frozen=True, slots=True)
class FieldSpec:
    wq_name: str
    category: FieldCategory
    source: FieldSource
    concept: str | None = None
    ticks_column: str | None = None
    depends_on: tuple[str, ...] = ()


def _build_registry() -> dict[str, FieldSpec]:
    fields: dict[str, FieldSpec] = {}

    def _pv(name: str, col: str) -> None:
        fields[name] = FieldSpec(
            wq_name=name,
            category=FieldCategory.PRICE_VOLUME,
            source=FieldSource.TICKS,
            ticks_column=col,
        )

    def _pv_computed(name: str, deps: tuple[str, ...]) -> None:
        fields[name] = FieldSpec(
            wq_name=name,
            category=FieldCategory.PRICE_VOLUME,
            source=FieldSource.COMPUTED,
            depends_on=deps,
        )

    def _fnd(name: str, concept: str | None = None) -> None:
        fields[name] = FieldSpec(
            wq_name=name,
            category=FieldCategory.FUNDAMENTAL,
            source=FieldSource.FUNDAMENTAL_RAW,
            concept=concept or name,
        )

    def _derived(name: str, deps: tuple[str, ...]) -> None:
        fields[name] = FieldSpec(
            wq_name=name,
            category=FieldCategory.DERIVED,
            source=FieldSource.COMPUTED,
            depends_on=deps,
        )

    def _group(name: str) -> None:
        fields[name] = FieldSpec(
            wq_name=name,
            category=FieldCategory.GROUP,
            source=FieldSource.METADATA,
        )

    # ── Price / Volume (direct ticks columns) ──
    _pv("close", "close")
    _pv("open", "open")
    _pv("high", "high")
    _pv("low", "low")
    _pv("volume", "volume")
    _pv("vwap", "vwap")

    # ── Price / Volume (computed) ──
    _pv_computed("returns", ("close",))
    _pv_computed("adv20", ("volume",))
    _pv_computed("cap", ("close", "sharesout"))
    _pv_computed("split", ())  # stub: 1.0

    # ── Fundamental (raw concepts from sec_mapping.yaml) ──
    _fnd("sharesout")
    _fnd("dividend")
    _fnd("assets")
    _fnd("liabilities")
    _fnd("operating_income")
    _fnd("sales")
    _fnd("capex")
    _fnd("equity")
    _fnd("debt_lt")
    _fnd("assets_curr")
    _fnd("goodwill")
    _fnd("income")
    _fnd("revenue")  # alias → same XBRL tags as sales
    _fnd("cashflow_op")
    _fnd("cash")
    _fnd("cogs")
    _fnd("liabilities_curr")
    _fnd("debt_st")
    _fnd("ppent")
    _fnd("cashflow", "cashflow")  # alias of cashflow_op
    _fnd("inventory")
    _fnd("cash_st", "cash_st")  # alias of cash
    _fnd("receivable")
    _fnd("sga_expense")
    _fnd("retained_earnings")
    _fnd("cashflow_fin")
    _fnd("income_tax")
    _fnd("pretax_income")
    _fnd("cashflow_invest", "cashflow_invest")
    _fnd("cashflow_dividends", "cashflow_dividends")  # alias of dividend
    _fnd("depre_amort")

    # ── Fundamental (new sec_mapping concepts) ──
    _fnd("fnd6_drft", "dr_lt")
    _fnd("fnd6_drc", "dr_st")
    _fnd("fnd6_ivaco", "invest_activity_other")
    _fnd("income_beforeextra", "income_before_extra")
    _fnd("fnd6_acdo", "assets_discontinued")
    _fnd("fnd6_clother", "other_income_nt")
    _fnd("fnd6_xrent", "rental_expense")

    # ── Derived (arithmetic on dependencies) ──
    _derived("debt", ("debt_lt", "debt_st"))
    _derived("invested_capital", ("equity", "debt_lt", "cash"))
    _derived("enterprise_value", ("cap", "debt_lt", "cash"))
    _derived("ebitda", ("operating_income", "depre_amort"))
    _derived("eps", ("income", "sharesout"))
    _derived("bookvalue_ps", ("equity", "sharesout"))
    _derived("operating_expense", ("cogs", "sga_expense", "depre_amort"))
    _derived("current_ratio", ("assets_curr", "liabilities_curr"))
    _derived("return_equity", ("income", "equity"))
    _derived("return_assets", ("income", "assets"))
    _derived("sales_ps", ("sales", "sharesout"))
    _derived("inventory_turnover", ("cogs", "inventory"))
    _derived("working_capital", ("assets_curr", "liabilities_curr"))
    _derived("sales_growth", ("sales",))

    # ── Group masks ──
    _group("sector")
    _group("industry")
    _group("subindustry")
    _group("exchange")

    return fields


ALL_FIELDS: dict[str, FieldSpec] = _build_registry()
VALID_FIELD_NAMES: frozenset[str] = frozenset(ALL_FIELDS)


def get_build_order() -> list[str]:
    """Return field names in dependency-safe build order (topological sort)."""
    graph: dict[str, set[str]] = {}
    for name, spec in ALL_FIELDS.items():
        graph[name] = set(spec.depends_on)
    sorter = TopologicalSorter(graph)
    return list(sorter.static_order())
