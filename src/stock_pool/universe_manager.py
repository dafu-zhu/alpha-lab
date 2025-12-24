from pathlib import Path
import shutil
import time
import polars as pl
from collection.ticks import Ticks
from stock_pool.universe import fetch_all_stocks

class UniverseManager:
    def __init__(self):
        self.recent_dir = Path("data/raw/ticks/recent")
        self.store_dir = Path("data/symbols")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.top_3000 = None

    def get_market_symbols(self, refresh=False) -> list[str]:
        """
        Get the official list of common stocks.
        """    
        csv_path = Path("data/symbols/stock_exchange.csv")
        if csv_path.exists() and not refresh:
            print("Loading symbols from local CSV...")
            df = pl.read_csv(csv_path)
            symbols = df["Ticker"].to_list()
        else:
            pd_df = fetch_all_stocks()
            if pd_df is not None and not pd_df.empty:
                symbols = pd_df['Ticker'].tolist()
            else:
                raise ValueError("Failed to fetch symbols from SEC.")
                
        print(f"Market Universe Size: {len(symbols)} tickers")
        return symbols
    
    def remove_recent_data(self) -> None:
        if self.recent_dir.exists():
            shutil.rmtree(self.recent_dir)

    def fetch_recent_data(self, symbols: list[str], refresh=False) -> None:
        """
        Downloads 3-month daily history for ALL symbols using the efficient bulk fetcher.
        """
        if refresh:
            self.remove_recent_data()

        print(f"Starting bulk fetch for {len(symbols)} symbols...")        
        # Instantiate Ticks with a dummy symbol (symbol arg is ignored for bulk fetch)
        fetcher = Ticks(symbol="UNIVERSE")
        
        # Call the bulk method 
        # Handles chunking (100 at a time), rate limits, and saving internally
        fetcher.fetch_and_store_bulk(symbols)
        
        print("Bulk fetch complete.")

    def filter_top_3000(self) -> list[str]:
        """
        Reads the cached data, calculates liquidity, and returns Top 3000.
        """       
        try:
            # Lazy Scan of all recent parquet files
            q = (
                pl.scan_parquet(self.recent_dir / "*.parquet")
                .group_by("symbol")
                .agg(
                    # Metric: Average Dollar Volume (Close * Volume)
                    (pl.col("close") * pl.col("volume")).mean().alias("avg_dollar_vol")
                )
                .filter(pl.col("avg_dollar_vol").is_not_null()) # Remove empty data
                .sort("avg_dollar_vol", descending=True)
                .head(3000)
            )
            df = q.collect()
            
            top_stock = df.row(0)
            bottom_stock = df.row(-1)
            print(f"Top Liquid Stock: {top_stock[0]} (ADV: ${top_stock[1]:,.0f})")
            print(f"Rank 3000 Stock:  {bottom_stock[0]} (ADV: ${bottom_stock[1]:,.0f})")
            
            result = df["symbol"].to_list()
            self.top_3000 = result
            return result
            
        except Exception as e:
            print(f"Error calculating liquidity: {e}")
            return []
    
    def store_top_3000(self) -> None:
        if not self.top_3000:
            self.top_3000 = self.filter_top_3000()
        
        file_path = self.store_dir / "universe_top3000.txt"
        with open(file_path, "w") as file:
            file.write("\n".join(self.top_3000))
        
        print(f"Saved Top 3000 symbols to {file_path}")

    def run(self, refresh=False) -> None:
        """
        Run complete pipeline
        Output universe_top3000.txt at /data/symbols
        """
        start = time.perf_counter()
        all_symbols = self.get_market_symbols(refresh=refresh)
        self.fetch_recent_data(all_symbols, refresh=refresh)
        self.store_top_3000()
        time_count = time.perf_counter() - start
        print(f"Processing time: {time_count:.2f}s")

if __name__ == "__main__":
    um = UniverseManager()
    um.run(refresh=True)