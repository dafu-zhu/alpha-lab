"""Unit tests for collection.alpaca_ticks module."""
import datetime as dt
import zoneinfo

import polars as pl
import pytest
from unittest.mock import Mock, patch

from quantdl.collection.alpaca_ticks import Ticks


MOCK_ENV = {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'}
MOCK_LOGGER_PATH = 'quantdl.collection.alpaca_ticks.LoggerFactory'
MOCK_REQUESTS_GET_PATH = 'quantdl.collection.alpaca_ticks.requests.get'

SAMPLE_BAR = {
    't': '2024-01-03T14:30:00Z',
    'o': 100.0, 'h': 101.0, 'l': 99.0, 'c': 100.5,
    'v': 1000000, 'n': 5000, 'vw': 100.25,
}


def _make_mock_logger(mock_logger_factory: Mock) -> Mock:
    """Wire up a mock LoggerFactory and return the mock logger."""
    mock_logger = Mock()
    mock_logger_factory.return_value.get_logger.return_value = mock_logger
    return mock_logger


def _make_api_response(bars: dict, next_page_token: str | None = None) -> Mock:
    """Build a mock Alpaca API response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        'bars': bars,
        'next_page_token': next_page_token,
    }
    return response


def _make_error_response(status_code: int = 500, text: str = 'Internal Server Error') -> Mock:
    """Build a mock error response."""
    response = Mock()
    response.status_code = status_code
    response.text = text
    return response


class TestTicks:
    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_initialization(self, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)

        ticks = Ticks()

        assert ticks.headers['accept'] == 'application/json'
        assert ticks.headers['APCA-API-KEY-ID'] == 'test_key'
        assert ticks.headers['APCA-API-SECRET-KEY'] == 'test_secret'
        mock_logger_factory.assert_called_once()

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_recent_daily_ticks_date_calculation(self, mock_get, mock_logger_factory):
        mock_logger = _make_mock_logger(mock_logger_factory)
        mock_get.return_value = _make_api_response({'AAPL': []})

        ticks = Ticks()
        ticks.recent_daily_ticks(symbols=['AAPL'], end_day='2024-06-30', window=90)

        log_calls = mock_logger.info.call_args_list
        assert len(log_calls) > 0
        assert '2024-06-30' in str(log_calls[0])


class TestParseTicks:
    def test_parse_ticks_basic(self):
        parsed = Ticks.parse_ticks([SAMPLE_BAR])

        assert len(parsed) == 1
        assert parsed[0].open == 100.0
        assert parsed[0].high == 101.0
        assert parsed[0].low == 99.0
        assert parsed[0].close == 100.5
        assert parsed[0].volume == 1000000
        assert parsed[0].num_trades == 5000
        assert parsed[0].vwap == 100.25

    def test_parse_ticks_timezone_conversion(self):
        """2024-01-03 14:30:00 UTC = 09:30:00 EST (UTC-5 during winter)."""
        parsed = Ticks.parse_ticks([SAMPLE_BAR])
        timestamp = dt.datetime.fromisoformat(parsed[0].timestamp)

        assert timestamp.hour == 9
        assert timestamp.minute == 30

    def test_parse_ticks_dst_conversion(self):
        """2024-06-15 13:30:00 UTC = 09:30:00 EDT (UTC-4 during summer)."""
        summer_bar = {**SAMPLE_BAR, 't': '2024-06-15T13:30:00Z'}
        parsed = Ticks.parse_ticks([summer_bar])
        timestamp = dt.datetime.fromisoformat(parsed[0].timestamp)

        assert timestamp.hour == 9
        assert timestamp.minute == 30

    def test_parse_ticks_multiple(self):
        second_bar = {
            't': '2024-01-03T14:31:00Z',
            'o': 100.5, 'h': 102.0, 'l': 100.0, 'c': 101.5,
            'v': 2000000, 'n': 6000, 'vw': 101.0,
        }
        parsed = Ticks.parse_ticks([SAMPLE_BAR, second_bar])

        assert len(parsed) == 2
        assert parsed[0].close == 100.5
        assert parsed[1].close == 101.5

    def test_parse_ticks_empty_list(self):
        assert Ticks.parse_ticks([]) == []

    def test_parse_ticks_invalid_data(self):
        with pytest.raises(ValueError, match="tick is empty"):
            Ticks.parse_ticks([None])


class TestGetMonthRange:
    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_january(self, mock_logger_factory):
        """ET boundaries: 4:00 AM Jan 1 to 8:00 PM Jan 31."""
        _make_mock_logger(mock_logger_factory)
        eastern = zoneinfo.ZoneInfo("America/New_York")

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 1)

        start_et = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00')).astimezone(eastern)
        end_et = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00')).astimezone(eastern)

        assert start_et.date() == dt.date(2024, 1, 1)
        assert start_et.hour == 4
        assert end_et.date() == dt.date(2024, 1, 31)
        assert end_et.hour == 20
        assert start_str.endswith('Z')
        assert end_str.endswith('Z')

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_february_leap_year(self, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)
        eastern = zoneinfo.ZoneInfo("America/New_York")

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 2)

        start_et = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00')).astimezone(eastern)
        end_et = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00')).astimezone(eastern)

        assert start_et.date() == dt.date(2024, 2, 1)
        assert end_et.date() == dt.date(2024, 2, 29)

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_december(self, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)
        eastern = zoneinfo.ZoneInfo("America/New_York")

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 12)

        start_et = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00')).astimezone(eastern)
        end_et = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00')).astimezone(eastern)

        assert start_et.date() == dt.date(2024, 12, 1)
        assert end_et.date() == dt.date(2024, 12, 31)


class TestRecentDailyTicks:
    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_success(self, mock_get, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)
        aapl_bar = {**SAMPLE_BAR, 't': '2024-06-28T04:00:00Z', 'o': 210.0, 'h': 212.0, 'l': 209.0, 'c': 211.0, 'v': 50000000, 'n': 200000, 'vw': 210.5}
        mock_get.return_value = _make_api_response({'AAPL': [aapl_bar]})

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert 'AAPL' in result
        assert isinstance(result['AAPL'], pl.DataFrame)
        assert len(result['AAPL']) == 1
        assert result['AAPL']['close'][0] == 211.0

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_pagination(self, mock_get, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)

        page1_bar = {'t': '2024-06-28T04:00:00Z', 'o': 210, 'h': 212, 'l': 209, 'c': 211, 'v': 50000000, 'n': 200000, 'vw': 210.5}
        page2_bar = {'t': '2024-06-29T04:00:00Z', 'o': 211, 'h': 213, 'l': 210, 'c': 212, 'v': 51000000, 'n': 210000, 'vw': 211.5}
        mock_get.side_effect = [
            _make_api_response({'AAPL': [page1_bar]}, next_page_token='token123'),
            _make_api_response({'AAPL': [page2_bar]}),
        ]

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert 'AAPL' in result
        assert len(result['AAPL']) == 2

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_api_error(self, mock_get, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)
        mock_get.return_value = _make_error_response()

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert result == {}

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_future_end_date_clamps_to_yesterday(self, mock_get, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)

        class FixedDateTime(dt.datetime):
            @classmethod
            def today(cls):
                return cls(2024, 6, 15, 12, 0, 0)

        mock_get.return_value = _make_api_response({'AAPL': []})

        with patch('quantdl.collection.alpaca_ticks.dt.datetime', FixedDateTime):
            ticks = Ticks()
            result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert isinstance(result, dict)

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_processing_error(self, mock_get, mock_logger_factory):
        mock_logger = _make_mock_logger(mock_logger_factory)
        invalid_bar = {**SAMPLE_BAR, 't': 'invalid-timestamp'}
        mock_get.return_value = _make_api_response({'AAPL': [invalid_bar]})

        ticks = Ticks()
        ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        mock_logger.error.assert_called()

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    @patch(MOCK_REQUESTS_GET_PATH)
    def test_request_exception(self, mock_get, mock_logger_factory):
        mock_logger = _make_mock_logger(mock_logger_factory)
        mock_get.side_effect = Exception("network failure")

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert result == {}
        mock_logger.error.assert_called()
        assert any("Request failed on page" in call[0][0] for call in mock_logger.error.call_args_list)


class TestFetchWithPagination:
    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_single_page(self, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)

        ticks = Ticks()
        mock_session = Mock()
        mock_session.get.return_value = _make_api_response({'AAPL': [SAMPLE_BAR]})

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01,
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 1

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_multiple_pages(self, mock_logger_factory):
        _make_mock_logger(mock_logger_factory)

        ticks = Ticks()
        mock_session = Mock()
        bar2 = {**SAMPLE_BAR, 't': '2024-01-03T14:31:00Z', 'o': 101, 'h': 102, 'l': 100, 'c': 101.5, 'v': 2000, 'n': 60, 'vw': 101.25}
        mock_session.get.side_effect = [
            _make_api_response({'AAPL': [SAMPLE_BAR]}, next_page_token='token123'),
            _make_api_response({'AAPL': [bar2]}),
        ]

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01,
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 2

    @patch.dict('os.environ', MOCK_ENV)
    @patch(MOCK_LOGGER_PATH)
    def test_api_error(self, mock_logger_factory):
        mock_logger = _make_mock_logger(mock_logger_factory)

        ticks = Ticks()
        mock_session = Mock()
        mock_session.get.return_value = _make_error_response()

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01,
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 0
        mock_logger.error.assert_called()
