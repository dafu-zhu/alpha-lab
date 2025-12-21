"""
Based on the stock list created from storage.stock_pool (data stored in data/symbols/stock_exchange.csv),
calculate the average dollar volume over the past 3 months of each stock as the liquidity score
"""
import os
import pandas as pd
import yfinance as yf
import datetime as dt
from multiprocessing import Pool, cpu_count
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

SYMBOL_PATH = os.path.join('data', 'symbols', 'stock_exchange.csv')
TICKS_DIR = os.path.join('data', 'ticks', 'daily')
OUTPUT_PATH = os.path.join('data', 'symbols', 'liquidity_scores.csv')

TEST_DATE = dt.datetime.today().date()
