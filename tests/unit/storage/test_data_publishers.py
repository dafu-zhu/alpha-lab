"""
Unit tests for storage.data_publishers module
Tests DataPublishers functionality with dependency injection
"""
from types import SimpleNamespace
from unittest.mock import Mock
import queue
import threading
import polars as pl


def _make_publisher():
    from quantdl.storage.data_publishers import DataPublishers

    s3_client = Mock()
    upload_config = SimpleNamespace(transfer={})
    logger = Mock()
    data_collectors = Mock()

    publisher = DataPublishers(
        s3_client=s3_client,
        upload_config=upload_config,
        logger=logger,
        data_collectors=data_collectors,
        bucket_name="test-bucket"
    )

    return publisher, s3_client, data_collectors


class TestDataPublishers:
    def test_initialization(self):
        publisher, s3_client, data_collectors = _make_publisher()

        assert publisher.s3_client == s3_client
        assert publisher.data_collectors == data_collectors
        assert publisher.bucket_name == "test-bucket"

    def test_publish_daily_ticks_success(self):
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, df)

        assert result == {"symbol": "AAPL", "status": "success", "error": None}
        s3_client.upload_fileobj.assert_called_once()

    def test_publish_daily_ticks_empty(self):
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, df)

        assert result["status"] == "skipped"
        s3_client.upload_fileobj.assert_not_called()

    def test_publish_fundamental_skips_without_cik(self):
        publisher, _, data_collectors = _make_publisher()

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik=None,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"
        data_collectors.collect_fundamental_long.assert_not_called()

    def test_publish_daily_ticks_value_error_skips(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=ValueError("not active on"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, df)

        assert result["status"] == "skipped"

    def test_publish_daily_ticks_unexpected_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, df)

        assert result["status"] == "failed"

    def test_publish_fundamental_success(self):
        publisher, _, data_collectors = _make_publisher()
        publisher.upload_fileobj = Mock()
        data_collectors._load_concepts.return_value = ["rev", "net_inc"]
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "as_of_date": ["2024-06-30", "2024-06-30"],
            "accn": ["1", "1"],
            "form": ["10-Q", "10-Q"],
            "concept": ["rev", "net_inc"],
            "value": [100.0, 10.0],
            "start": ["2024-04-01", "2024-04-01"],
            "end": ["2024-06-30", "2024-06-30"],
            "frame": ["CY2024Q2", "CY2024Q2"],
            "is_instant": [False, False]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_ttm_fundamental_success(self):
        publisher, _, data_collectors = _make_publisher()
        publisher.upload_fileobj = Mock()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.return_value = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["rev"],
            "value": [100.0]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_ttm_fundamental_empty(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.return_value = pl.DataFrame()

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"

    def test_publish_derived_fundamental_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_derived_fundamental_empty(self):
        publisher, _, _ = _make_publisher()
        derived_df = pl.DataFrame()

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df
        )

        assert result["status"] == "skipped"

    def test_publish_top_3000_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=["AAPL", "MSFT"],
            source="test"
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_top_3000_empty(self):
        publisher, _, _ = _make_publisher()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=[],
            source="test"
        )

        assert result["status"] == "skipped"

    def test_minute_ticks_worker_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()
        df = pl.DataFrame({
            "timestamp": ["2024-06-30T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })
        q.put(("AAPL", "2024-06-30", df))
        q.put(None)

        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()
        t.join()

        assert stats["success"] == 1
        publisher.upload_fileobj.assert_called_once()

    def test_minute_ticks_worker_skips_empty(self):
        publisher, _, _ = _make_publisher()
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()
        q.put(("AAPL", "2024-06-30", pl.DataFrame()))
        q.put(None)

        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()
        t.join()

        assert stats["skipped"] == 1
