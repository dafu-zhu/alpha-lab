"""Tests for feature field registry."""

import pytest


class TestFieldRegistry:
    def test_all_fields_populated(self):
        from quantdl.features.registry import ALL_FIELDS
        assert len(ALL_FIELDS) > 50

    def test_all_names_unique(self):
        from quantdl.features.registry import ALL_FIELDS
        assert len(ALL_FIELDS) == len(set(ALL_FIELDS.keys()))

    def test_valid_field_names_matches(self):
        from quantdl.features.registry import ALL_FIELDS, VALID_FIELD_NAMES
        assert VALID_FIELD_NAMES == frozenset(ALL_FIELDS.keys())

    def test_unknown_key_raises(self):
        from quantdl.features.registry import ALL_FIELDS
        with pytest.raises(KeyError):
            _ = ALL_FIELDS["nonexistent_field_xyz"]

    def test_close_is_price_volume(self):
        from quantdl.features.registry import ALL_FIELDS, FieldCategory, FieldSource
        spec = ALL_FIELDS["close"]
        assert spec.category == FieldCategory.PRICE_VOLUME
        assert spec.source == FieldSource.TICKS
        assert spec.ticks_column == "close"

    def test_assets_is_fundamental(self):
        from quantdl.features.registry import ALL_FIELDS, FieldCategory, FieldSource
        spec = ALL_FIELDS["assets"]
        assert spec.category == FieldCategory.FUNDAMENTAL
        assert spec.source == FieldSource.FUNDAMENTAL_RAW
        assert spec.concept == "assets"

    def test_debt_is_derived(self):
        from quantdl.features.registry import ALL_FIELDS, FieldCategory
        spec = ALL_FIELDS["debt"]
        assert spec.category == FieldCategory.DERIVED
        assert "debt_lt" in spec.depends_on
        assert "debt_st" in spec.depends_on

    def test_sector_is_group(self):
        from quantdl.features.registry import ALL_FIELDS, FieldCategory, FieldSource
        spec = ALL_FIELDS["sector"]
        assert spec.category == FieldCategory.GROUP
        assert spec.source == FieldSource.METADATA

    def test_returns_depends_on_close(self):
        from quantdl.features.registry import ALL_FIELDS
        spec = ALL_FIELDS["returns"]
        assert "close" in spec.depends_on

    def test_concept_mapping_cashflow_invest(self):
        from quantdl.features.registry import ALL_FIELDS
        spec = ALL_FIELDS["cashflow_invest"]
        assert spec.concept == "cashflow_invest"

    def test_fnd6_drft_concept(self):
        from quantdl.features.registry import ALL_FIELDS
        spec = ALL_FIELDS["fnd6_drft"]
        assert spec.concept == "dr_lt"


class TestBuildOrder:
    def test_build_order_returns_list(self):
        from quantdl.features.registry import get_build_order
        order = get_build_order()
        assert isinstance(order, list)
        assert len(order) > 0

    def test_build_order_contains_all_fields(self):
        from quantdl.features.registry import get_build_order, ALL_FIELDS
        order = get_build_order()
        assert set(order) == set(ALL_FIELDS.keys())

    def test_dependencies_before_dependents(self):
        from quantdl.features.registry import get_build_order, ALL_FIELDS
        order = get_build_order()
        positions = {name: i for i, name in enumerate(order)}
        for name, spec in ALL_FIELDS.items():
            for dep in spec.depends_on:
                assert positions[dep] < positions[name], (
                    f"{dep} must come before {name} in build order"
                )

    def test_close_before_returns(self):
        from quantdl.features.registry import get_build_order
        order = get_build_order()
        assert order.index("close") < order.index("returns")

    def test_sharesout_before_eps(self):
        from quantdl.features.registry import get_build_order
        order = get_build_order()
        assert order.index("sharesout") < order.index("eps")
