import requests
import time
from collection.models import FndDataPoint
import datetime as dt
from typing import List, Optional
from pathlib import Path
from collections import defaultdict
import polars as pl
import json

HEADER = {'User-Agent': 'name@example.com'}

class Fundamental:
    def __init__(self, cik: str, symbol: Optional[str] = None) -> None:
        self.cik = cik
        self.symbol = symbol
        self.log_dir = Path("data/logs/fundamental")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_facts(self, field: str, sleep=True) -> List[dict]:
        """
        Get historical facts of a company using SEC XBRL

        :param cik: Company identifier
        :type cik: str
        :param field: Accounting data to fetch
        :type field: str
        :raises requests.RequestException: If HTTP request fails
        :raises KeyError: If field is not available for this company
        :raises ValueError: If data format is unexpected
        """
        cik_padded = str(self.cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        try:
            response = requests.get(url=url, headers=HEADER)
            response.raise_for_status()
            res = response.json()
        except requests.RequestException as error:
            raise requests.RequestException(f"Failed to fetch data for CIK {cik_padded}: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON response for CIK {cik_padded}: {error}")

        # Avoid reaching api rate limit (10/s)
        if sleep:
            time.sleep(0.1)

        # Check if facts and us-gaap exist
        if 'facts' not in res:
            raise KeyError(f"No 'facts' data found for CIK {cik_padded}")

        if 'us-gaap' not in res['facts']:
            raise KeyError(f"No 'us-gaap' data found for CIK {cik_padded}")

        gaap = res['facts']['us-gaap']

        # Check if field exists
        if field not in gaap:
            available_fields = list(gaap.keys())
            raise KeyError(
                f"Field '{field}' not available for CIK {cik_padded}. "
                f"Available fields: {len(available_fields)} total"
            )

        # Check if USD units exist for this field
        if 'units' not in gaap[field]:
            raise KeyError(f"No 'units' data found for field '{field}' in CIK {cik_padded}")

        if 'USD' not in gaap[field]['units']:
            available_units = list(gaap[field]['units'].keys())
            raise KeyError(
                f"USD units not available for field '{field}' in CIK {cik_padded}. "
                f"Available units: {available_units}"
            )

        usd_result = gaap[field]['units']['USD']

        return usd_result

    def get_dps(self, field: str) -> List[FndDataPoint]:
        """
        Transform raw data point into FndDataPoint object
        """
        raw_data = self.get_facts(field)
        
        dps = []
        for dp in raw_data:
            # Reveal date
            filed_date = dt.datetime.strptime(dp['filed'], '%Y-%m-%d').date()
            
            # Fiscal calendar date, avoid look-ahead bias
            end_date = dt.datetime.strptime(dp['end'], '%Y-%m-%d').date()
            
            # Form to track amendment
            form = dp['form']

            dp_obj = FndDataPoint(
                timestamp=filed_date,
                value=dp['val'],
                end_date=end_date,
                fy=dp['fy'],
                fp=dp['fp'],
                form=form
            )
            dps.append(dp_obj)

        return dps

    def _log_error(self, field: str, error_type: str, error_message: str) -> None:
        """
        Log field fetching errors to JSON file.

        :param field: XBRL field name that failed
        :param error_type: Type of error (e.g., 'FieldNotAvailable', 'RequestException')
        :param error_message: Detailed error message
        """
        log_entry = {
            'timestamp': dt.datetime.now().isoformat(),
            'cik': self.cik,
            'symbol': self.symbol or 'UNKNOWN',
            'field': field,
            'error_type': error_type,
            'error_message': error_message
        }

        # Create log file path with current date
        log_date = dt.datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"errors_{log_date}.json"

        # Read existing logs or create new list
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        # Append new log entry
        logs.append(log_entry)

        # Write back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def _deduplicate_dps(self, dps: List[FndDataPoint]) -> List[FndDataPoint]:
        """
        Deduplicate datapoints by keeping the most recent filing per fiscal period.

        Groups by (end_date, fy, fp) and selects the filing with the latest timestamp.
        This ensures amendments (10-K/A) supersede original filings (10-K).

        :param dps: List of FundamentalDataPoint objects
        :return: Deduplicated list of FundamentalDataPoint objects
        """
        # Group by fiscal period: (end_date, fy, fp)
        groups = defaultdict(list)
        for dp in dps:
            key = (dp.end_date, dp.fy, dp.fp)
            groups[key].append(dp)

        # Select most recent filing per group
        deduplicated = []
        for key, group in groups.items():
            # Sort by timestamp (filed date) descending, take the most recent
            most_recent = max(group, key=lambda x: x.timestamp)
            deduplicated.append(most_recent)

        # Sort by timestamp for chronological order
        deduplicated.sort(key=lambda x: x.timestamp)

        return deduplicated

    def _generate_field_daily_values(self, year: int, field: str) -> Optional[List]:
        """
        Generate daily values for a single field for a given year.

        For every calendar day in the year, assigns the most recent filed value
        as of that date (forward-fill logic).

        :param year: Year to generate data for (e.g., 2024)
        :param field: XBRL field name (e.g., 'CashAndCashEquivalentsAtCarryingValue')
        :return: List of daily values (one per day of the year), or None if field unavailable
        """
        try:
            # Get and deduplicate datapoints
            raw_dps = self.get_dps(field)
            dps = self._deduplicate_dps(raw_dps)

            # Generate daily values for the year
            start_date = dt.date(year, 1, 1)
            end_date = dt.date(year, 12, 31)

            values = []
            current_value = None
            current_day = start_date
            dp_index = 0

            while current_day <= end_date:
                # Update current_value if a new filing was released on or before this day
                while dp_index < len(dps) and dps[dp_index].timestamp <= current_day:
                    current_value = dps[dp_index].value
                    dp_index += 1

                # Assign value for this day (None if no filing has been released yet)
                values.append(current_value)
                current_day += dt.timedelta(days=1)

            return values

        except KeyError as e:
            # Field not available for this company
            self._log_error(field, 'FieldNotAvailable', str(e))
            print(f"  ⚠ Field '{field}' not available (logged)")
            return None

        except requests.RequestException as e:
            # Network or API error
            self._log_error(field, 'RequestException', str(e))
            print(f"  ⚠ Request failed for '{field}' (logged)")
            return None

        except Exception as e:
            # Unexpected error
            self._log_error(field, 'UnexpectedException', str(e))
            print(f"  ⚠ Unexpected error for '{field}': {e} (logged)")
            return None

    def generate_year_data(self, year: int, fields: List[str], symbol: str) -> None:
        """
        Generate daily values for a given year and multiple fields, save to Parquet.

        For every calendar day in the year, assigns the most recent filed value
        as of that date (forward-fill logic) for each field.

        :param year: Year to generate data for (e.g., 2024)
        :param fields: List of XBRL field names (e.g., ['CashAndCashEquivalentsAtCarryingValue', 'Assets'])
        :param symbol: Stock symbol (e.g., 'AAPL')
        """
        # Update symbol if not set during initialization
        if not self.symbol:
            self.symbol = symbol

        # Generate date range for the year
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        dates = []
        current_day = start_date
        while current_day <= end_date:
            dates.append(current_day.isoformat())
            current_day += dt.timedelta(days=1)

        # Create DataFrame starting with date column
        data = {'date': dates}
        successful_fields = 0
        failed_fields = 0

        # Process each field
        for field in fields:
            print(f"Processing field: {field}")
            values = self._generate_field_daily_values(year, field)

            if values is not None:
                data[field] = values
                successful_fields += 1
            else:
                # Add None values for this field if it fails
                data[field] = [None] * len(dates)
                failed_fields += 1

        # Create Polars DataFrame
        df = pl.DataFrame(data)

        # Save to Parquet
        output_dir = Path(f"data/fundamental/{symbol}/{year}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "fundamental.parquet"
        df.write_parquet(output_file, compression='zstd')

        print(f"\n✓ Saved fundamental data for {symbol} ({year})")
        print(f"  Successfully processed: {successful_fields}/{len(fields)} fields")
        if failed_fields > 0:
            print(f"  Failed fields: {failed_fields} (see logs in {self.log_dir})")
        print(f"  Output: {output_file}")





# Example usage
if __name__ == "__main__":
    cik = '1819994'  # Rocket Lab USA Inc.
    symbol = 'RKLB'
    fields = [
        'CostOfGoodsAndServicesSold',  # May not be available for all companies
        'Assets',
        'Liabilities',
        'StockholdersEquity',
        'Revenues',  # Alternative field names to test
        'CashAndCashEquivalentsAtCarryingValue'
    ]
    year = 2024

    # Create Fundamentals instance with symbol for better logging
    fund = Fundamental(cik, symbol=symbol)

    # Generate and save year data for all fields
    print(f"Generating daily data for {symbol} {year} with {len(fields)} fields...")
    print("=" * 60)
    fund.generate_year_data(year=year, fields=fields, symbol=symbol)
