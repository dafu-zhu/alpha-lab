"""Storage client implementations."""

from alphalab.data.storage.local import LocalStorageClient, StreamingBody

StorageClient = LocalStorageClient

__all__ = [
    "StorageClient",
    "LocalStorageClient",
    "StreamingBody",
    "TicksClient",
]


def __getattr__(name: str):
    """Lazy import for TicksClient to avoid circular import."""
    if name == "TicksClient":
        from alphalab.data.storage.ticks import TicksClient
        return TicksClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
