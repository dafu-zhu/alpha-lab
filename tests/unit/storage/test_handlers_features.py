"""Tests for FeaturesHandler."""

import pytest
from datetime import date
from unittest.mock import Mock, MagicMock


class TestFeaturesHandler:
    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies for FeaturesHandler."""
        feature_builder = Mock()
        feature_builder.build_all.return_value = {"close": "/tmp/close.arrow"}

        security_master = Mock()
        security_master.get_security_id.return_value = 12345

        universe_manager = Mock()
        universe_manager.load_symbols_for_year.return_value = ["AAPL", "MSFT"]

        calendar = Mock()
        calendar.get_trading_days.return_value = [
            date(2024, 1, 2), date(2024, 1, 3),
        ]

        logger = Mock()

        return feature_builder, security_master, universe_manager, calendar, logger

    def test_build_calls_feature_builder(self, mock_deps):
        from quantdl.storage.handlers.features import FeaturesHandler

        fb, sm, um, cal, log = mock_deps
        handler = FeaturesHandler(fb, sm, um, cal, log)
        result = handler.build(2024, 2024)

        fb.build_all.assert_called_once()
        assert "close" in result

    def test_build_collects_security_ids(self, mock_deps):
        from quantdl.storage.handlers.features import FeaturesHandler

        fb, sm, um, cal, log = mock_deps
        handler = FeaturesHandler(fb, sm, um, cal, log)
        handler.build(2024, 2024)

        # Should have called get_security_id for each symbol
        assert sm.get_security_id.call_count == 2

    def test_build_skips_unresolvable_symbols(self, mock_deps):
        from quantdl.storage.handlers.features import FeaturesHandler

        fb, sm, um, cal, log = mock_deps
        sm.get_security_id.side_effect = [12345, ValueError("not found")]

        handler = FeaturesHandler(fb, sm, um, cal, log)
        handler.build(2024, 2024)

        # Should still call build_all with the one resolved sid
        call_args = fb.build_all.call_args
        sids = call_args.kwargs.get("security_ids") or call_args[1].get("security_ids") or call_args[0][1]
        assert "12345" in sids

    def test_build_multi_year(self, mock_deps):
        from quantdl.storage.handlers.features import FeaturesHandler

        fb, sm, um, cal, log = mock_deps
        handler = FeaturesHandler(fb, sm, um, cal, log)
        handler.build(2023, 2024)

        # Should load symbols for both years
        assert um.load_symbols_for_year.call_count == 2

    def test_build_passes_overwrite(self, mock_deps):
        from quantdl.storage.handlers.features import FeaturesHandler

        fb, sm, um, cal, log = mock_deps
        handler = FeaturesHandler(fb, sm, um, cal, log)
        handler.build(2024, 2024, overwrite=True)

        call_kwargs = fb.build_all.call_args.kwargs
        assert call_kwargs.get("overwrite") is True
