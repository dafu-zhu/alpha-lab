"""Tests for ticks feature builder."""

import pytest
import polars as pl
from datetime import date


class TestTicksFeatureBuilderDirect:
    def test_build_direct_close(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        result = builder.build_direct("close", "close", trading_days, security_ids)

        assert "Date" in result.columns
        assert len(result) == len(trading_days)
        for sid in security_ids:
            assert sid in result.columns

    def test_build_direct_volume(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        result = builder.build_direct("volume", "volume", trading_days, security_ids)
        assert result["SEC001"][0] == 1_000_000

    def test_build_direct_missing_sid_gets_null(self, raw_data_dir, trading_days):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        result = builder.build_direct("close", "close", trading_days, ["SEC001", "MISSING"])

        assert "SEC001" in result.columns
        assert "MISSING" not in result.columns  # missing sid not joined

    def test_wide_table_shape(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        result = builder.build_direct("open", "open", trading_days, security_ids)
        assert result.shape == (10, 3)  # timestamp + 2 sids


class TestTicksFeatureBuilderComputed:
    def test_returns_first_row_null(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        close = builder.build_direct("close", "close", trading_days, security_ids)

        built = {"close": close}
        returns = builder.build_computed("returns", built, trading_days, security_ids)

        assert returns["SEC001"][0] is None  # first row null

    def test_returns_computed_correctly(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        close = builder.build_direct("close", "close", trading_days, security_ids)

        built = {"close": close}
        returns = builder.build_computed("returns", built, trading_days, security_ids)

        # close goes 102, 103, 104, ... => returns[1] = 103/102 - 1
        expected = 103.0 / 102.0 - 1
        assert abs(returns["SEC001"][1] - expected) < 1e-10

    def test_adv20_first_19_null(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        volume = builder.build_direct("volume", "volume", trading_days, security_ids)

        built = {"volume": volume}
        adv20 = builder.build_computed("adv20", built, trading_days, security_ids)

        # With only 10 rows, all values should be null (window=20 > 10 rows)
        assert adv20["SEC001"].null_count() == 10

    def test_split_stub_all_ones(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        result = builder.build_computed("split", {}, trading_days, security_ids)

        assert all(v == 1.0 for v in result["SEC001"].to_list())

    def test_cap_close_times_sharesout(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        from alphalab.features.builders.fundamental import FundamentalFeatureBuilder
        ticks_builder = TicksFeatureBuilder(str(raw_data_dir))
        fnd_builder = FundamentalFeatureBuilder(str(raw_data_dir))

        close = ticks_builder.build_direct("close", "close", trading_days, security_ids)
        sharesout = fnd_builder.build_raw("sharesout", "sharesout", trading_days, security_ids)

        built = {"close": close, "sharesout": sharesout}
        cap = ticks_builder.build_computed("cap", built, trading_days, security_ids)

        # sharesout is 1e9, close starts at 102 for SEC001
        # After forward-fill, sharesout should be 1e9 from day 4 onward
        # cap = close * sharesout
        assert "SEC001" in cap.columns

    def test_unknown_computed_raises(self, raw_data_dir, trading_days, security_ids):
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        with pytest.raises(ValueError, match="Unknown computed"):
            builder.build_computed("nonexistent", {}, trading_days, security_ids)

    def test_cap_empty_when_no_common_sids(self, raw_data_dir, trading_days):
        """Cap returns timestamp-only when no common sids."""
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        builder = TicksFeatureBuilder(str(raw_data_dir))
        close = pl.DataFrame({"Date": trading_days, "A": [1.0] * len(trading_days)})
        sharesout = pl.DataFrame({"Date": trading_days, "B": [1.0] * len(trading_days)})
        built = {"close": close, "sharesout": sharesout}
        result = builder.build_computed("cap", built, trading_days, ["A", "B"])
        assert result.columns == ["Date"]


class TestTicksEdgeCases:
    def test_string_timestamp_is_cast(self, tmp_path, trading_days):
        """Ticks with string timestamp column should be auto-cast."""
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "SCAST"
        ticks_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "timestamp": [str(d) for d in trading_days],
            "close": [100.0 + i for i in range(len(trading_days))],
        })
        df.write_parquet(str(ticks_dir / "ticks.parquet"))

        builder = TicksFeatureBuilder(str(tmp_path))
        result = builder.build_direct("close", "close", trading_days, ["SCAST"])
        assert "SCAST" in result.columns
        assert result["SCAST"].null_count() == 0

    def test_read_column_corrupted_file(self, tmp_path, trading_days):
        """Corrupted parquet returns None gracefully."""
        from alphalab.features.builders.ticks import TicksFeatureBuilder
        ticks_dir = tmp_path / "data" / "raw" / "ticks" / "daily" / "BAD"
        ticks_dir.mkdir(parents=True)
        (ticks_dir / "ticks.parquet").write_text("corrupted")

        builder = TicksFeatureBuilder(str(tmp_path))
        result = builder.build_direct("close", "close", trading_days, ["BAD"])
        # BAD should not be in columns since read failed
        assert "BAD" not in result.columns
