"""
Storage client implementations.
"""

from quantdl.storage.clients.s3 import S3Client
from quantdl.storage.clients.local import LocalStorageClient, StreamingBody

# TicksClient uses lazy import to avoid circular dependency with master.security_master
# Use: from quantdl.storage.clients import TicksClient
# Or: from quantdl.storage.clients.ticks import TicksClient

__all__ = [
    'S3Client',
    'LocalStorageClient',
    'StreamingBody',
    'TicksClient',
]


def __getattr__(name: str):
    """Lazy import for TicksClient to avoid circular import."""
    if name == 'TicksClient':
        from quantdl.storage.clients.ticks import TicksClient
        return TicksClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
