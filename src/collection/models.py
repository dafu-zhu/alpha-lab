from dataclasses import dataclass
import datetime
from typing import Dict
from enum import Enum

@dataclass
class FndDataPoint:
    timestamp: datetime.date
    value: float
    end_date: datetime.date
    fy: int
    fp: str
    form: str

@dataclass
class TickDataPoint:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    num_trades: int
    vwap: float

class TickField(Enum):
    CLOSE = 'c'
    HIGH = 'h'
    LOW = 'l'
    NUM_TRADES = 'n'
    OPEN = 'o'
    TIMESTAMP = 't'
    VOLUME = 'v'
    VWAP = 'vw'