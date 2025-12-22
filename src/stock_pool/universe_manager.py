from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm

from collection.ticks import Ticks

from stock_pool.universe import fetch_sec_stocks

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
            pd_df = fetch_sec_stocks()
            if pd_df is not None and not pd_df.empty:
                symbols = pd_df['Ticker'].tolist()
            else:
                raise ValueError("Failed to fetch symbols from SEC.")
                
        print(f"Market Universe Size: {len(symbols)} tickers")
        return symbols

    def fetch_recent_data(self, symbols: list[str]) -> None:
        """
        Downloads 3-month daily history for ALL symbols in parallel.
        """

        # Alpaca allows ~200 req/min. 
        # Ticks.get_ticks has a 0.1s sleep, limiting to 10/sec per thread.
        # 5 threads = 50 req/sec (too fast). 2 threads = 20 req/sec (safe).
        # Use 4 workers and rely on the internal sleep.
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Map symbol -> Future
            future_to_symbol = {
                executor.submit(Ticks(sym).update_liquidity_cache): sym 
                for sym in symbols
            }
            
            # Progress bar
            for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Caching Data"):
                pass 

    def filter_top_3000(self) -> list[str]:
        """
        Reads the cached data, calculates liquidity, and returns Top 3000.
        """       
        try:
            # 1. Lazy Scan of all recent parquet files
            # This is extremely fast (Polars is built for this)
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
            
            # 2. Execute
            df = q.collect()
            
            # 3. Output stats
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
        all_symbols = self.get_market_symbols(refresh=refresh)
        self.fetch_recent_data(all_symbols)
        self.store_top_3000()


if __name__ == "__main__":
    um = UniverseManager()
    um.run()