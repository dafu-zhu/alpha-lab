"""Tests for storage.clients.__init__ lazy import."""

import pytest


class TestLazyImport:
    def test_ticks_client_lazy_import(self):
        """TicksClient is importable via lazy __getattr__."""
        from alphalab.storage.clients import TicksClient
        assert TicksClient is not None

    def test_invalid_attribute_raises(self):
        """Accessing invalid attribute raises AttributeError."""
        import alphalab.storage.clients as clients
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = clients.NonexistentClass

    def test_exports(self):
        """__all__ contains expected exports."""
        from alphalab.storage.clients import __all__
        assert "StorageClient" in __all__
        assert "LocalStorageClient" in __all__
        assert "TicksClient" in __all__
