# Concurrency Tutorial for Data Lake Operations

## Introduction

This tutorial teaches you how to use **concurrency** to speed up data collection in the US Equity Data Lake. We'll go from sequential (slow) to parallel (fast) operations, explaining each concept step by step.

### What You'll Learn

- **Why concurrency matters**: 21 hours → 1 hour for data collection
- **Threading vs Multiprocessing**: Which to use when
- **ThreadPoolExecutor**: The core tool for parallel I/O operations
- **Practical patterns**: Upload files, fetch APIs, process data in parallel
- **Common pitfalls**: Thread safety, shared state, error handling

### Prerequisites

- Python 3.8+
- Basic understanding of functions and loops
- Familiarity with boto3 S3 operations

---

## Table of Contents

1. [Why Concurrency Matters](#1-why-concurrency-matters)
2. [Threading vs Multiprocessing](#2-threading-vs-multiprocessing)
3. [ThreadPoolExecutor - Core Concepts](#3-threadpoolexecutor---core-concepts)
4. [Pattern 1: Parallel File Uploads](#4-pattern-1-parallel-file-uploads)
5. [Pattern 2: Parallel API Calls](#5-pattern-2-parallel-api-calls)
6. [Pattern 3: Parallel Processing in Lambda](#6-pattern-3-parallel-processing-in-lambda)
7. [Thread Safety and Shared State](#7-thread-safety-and-shared-state)
8. [Error Handling in Parallel Operations](#8-error-handling-in-parallel-operations)
9. [Complete Integration Example](#9-complete-integration-example)

---

## 1. Why Concurrency Matters

### The Problem: Sequential Processing is Slow

**Scenario:** Upload daily tick data for 5,000 symbols (75,000 files total).

**Sequential approach (one at a time):**

```python
import time

def upload_file(symbol):
    """Upload takes ~1 second per file."""
    time.sleep(1)  # Simulates S3 upload
    print(f"Uploaded {symbol}")

# Sequential: Process one symbol at a time
start = time.time()
symbols = ['AAPL', 'MSFT', 'GOOGL', ...]  # 5,000 symbols

for symbol in symbols:
    upload_file(symbol)

duration = time.time() - start
print(f"Duration: {duration} seconds")
# Output: Duration: 5000 seconds (~1.4 hours)
```

**Why it's slow:**
```
CPU: [Upload AAPL] [Wait...] [Upload MSFT] [Wait...] [Upload GOOGL] [Wait...]
     └─ Active ─┘  └─ Idle ─┘ └─ Active ─┘  └─ Idle ─┘

Result: 99% of time spent WAITING for S3, not working!
```

### The Solution: Parallel Processing

**Parallel approach (20 files at once):**

```python
from concurrent.futures import ThreadPoolExecutor

# Parallel: Process 20 symbols simultaneously
start = time.time()

with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(upload_file, symbols)

duration = time.time() - start
print(f"Duration: {duration} seconds")
# Output: Duration: 250 seconds (~4 minutes)
```

**Why it's fast:**
```
Thread 1: [Upload AAPL] [Wait...] [Upload TSLA] [Wait...]
Thread 2: [Upload MSFT] [Wait...] [Upload NVDA] [Wait...]
Thread 3: [Upload GOOGL] [Wait...] [Upload AMD] [Wait...]
...
Thread 20: [Upload AMZN] [Wait...] [Upload NFLX] [Wait...]

Result: 20 uploads happening simultaneously = 20x faster!
```

### Performance Comparison

| Approach | Duration | Speedup |
|----------|----------|---------|
| Sequential (1 at a time) | 5,000 seconds (~1.4 hours) | 1x (baseline) |
| Parallel (20 workers) | 250 seconds (~4 minutes) | **20x faster** |
| Parallel (50 workers) | 100 seconds (~1.7 minutes) | **50x faster** |

**For 75,000 files (15 years × 5,000 symbols):**
- Sequential: ~21 hours
- Parallel (20 workers): ~1 hour

---

## 2. Threading vs Multiprocessing

### Understanding Python Concurrency

Python has two main approaches to concurrency:

#### Threading: For I/O-Bound Tasks

**What it is:**
- Multiple threads share the same process
- Great for waiting (I/O operations)
- Limited by GIL (Global Interpreter Lock) for CPU tasks

**When to use:**
- ✅ S3 uploads/downloads (waiting for network)
- ✅ API calls to Alpaca/SEC (waiting for responses)
- ✅ Database queries (waiting for DB)
- ✅ File I/O (waiting for disk)

**Why it works:**
```python
# Thread 1 uploads file, waits for S3 response
# While waiting, thread 1 releases GIL
# Thread 2 can now use CPU to start its upload
# Both threads spend 99% time waiting, 1% working
# GIL doesn't matter because we're mostly waiting!
```

#### Multiprocessing: For CPU-Bound Tasks

**What it is:**
- Multiple separate processes
- Each has its own Python interpreter
- No GIL limitations
- Higher memory overhead

**When to use:**
- ✅ Data compression/decompression
- ✅ Complex calculations (ML, statistics)
- ✅ Image/video processing
- ❌ NOT for S3 uploads (threading is better)

**Why we don't use it for data lake:**
```python
# Multiprocessing overhead:
# - Start new process: ~100ms
# - Serialize data between processes: ~10ms
# - Higher memory usage: 50MB+ per process

# For I/O tasks like S3 uploads:
# - Threading overhead: ~1ms
# - No serialization needed
# - Low memory: ~1MB per thread

# For 5,000 uploads: Threading wins!
```

### Decision Tree

```
Is your task waiting for external systems? (S3, API, DB)
├─ YES → Use Threading (ThreadPoolExecutor)
└─ NO → Is it CPU-intensive? (compression, math)
    ├─ YES → Use Multiprocessing (ProcessPoolExecutor)
    └─ NO → Sequential is fine
```

**For US Equity Data Lake:**
- Daily tick collection: **Threading** (API calls to Alpaca)
- S3 uploads: **Threading** (waiting for network)
- Data parsing: **Threading** (JSON/Parquet I/O is fast enough)
- Future ML models: **Multiprocessing** (CPU-heavy calculations)

---

## 3. ThreadPoolExecutor - Core Concepts

### What is ThreadPoolExecutor?

A high-level interface for running tasks in parallel threads. Think of it as a **team of workers** that process jobs from a queue.

```
┌─────────────────────────────────────────┐
│      ThreadPoolExecutor (max_workers=3) │
├─────────────────────────────────────────┤
│  Worker 1: [Processing task...]         │
│  Worker 2: [Processing task...]         │
│  Worker 3: [Processing task...]         │
└─────────────────────────────────────────┘
           ↑
           │
    Task Queue: [Task 4, Task 5, Task 6, ...]
```

### Core Components

#### 1. Creating the Executor

```python
from concurrent.futures import ThreadPoolExecutor

# Create executor with 20 workers
executor = ThreadPoolExecutor(max_workers=20)

# Best practice: Use context manager (auto cleanup)
with ThreadPoolExecutor(max_workers=20) as executor:
    # Submit tasks here
    pass  # Executor automatically shuts down when done
```

**Parameters:**

| Parameter | Type | Purpose | Our Setting |
|-----------|------|---------|-------------|
| `max_workers` | int | Number of parallel threads | **20** (for S3 uploads) |
| `thread_name_prefix` | str | Prefix for thread names (debugging) | Optional |
| `initializer` | callable | Function run when thread starts | Rarely used |

**Why max_workers=20?**
```python
# Too few (5 workers):
# - Underutilized: Only 5 uploads at once
# - Slower: Takes longer to finish

# Just right (20 workers):
# - Good balance: 20 parallel uploads
# - Fast enough: Completes in reasonable time
# - Not overwhelming: S3 can handle 20 connections

# Too many (200 workers):
# - Diminishing returns: S3 rate limits kick in
# - High overhead: Thread switching costs
# - Connection errors: Too many simultaneous connections
```

#### 2. Submitting Tasks

**Method 1: `submit()` - For single tasks**

```python
def upload_file(symbol, year):
    """Upload one file to S3."""
    # Upload logic here
    return f"Uploaded {symbol} {year}"

# Submit a single task
with ThreadPoolExecutor(max_workers=20) as executor:
    future = executor.submit(upload_file, 'AAPL', 2024)
    result = future.result()  # Wait for completion
    print(result)  # Output: "Uploaded AAPL 2024"
```

**What happens:**
```
1. executor.submit() → Adds task to queue, returns Future object
2. Worker thread picks up task from queue
3. Worker executes upload_file('AAPL', 2024)
4. future.result() → Waits for completion, returns result
```

**Method 2: `map()` - For multiple tasks (same function)**

```python
# Upload multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(upload_file, symbols)

    # Results are in same order as input
    for result in results:
        print(result)
```

**What happens:**
```
1. executor.map() → Submits 5 tasks to queue
2. 5 worker threads process tasks in parallel
3. Results returned in original order (AAPL, MSFT, GOOGL, ...)
```

#### 3. Retrieving Results

**Pattern 1: Wait for all tasks to complete**

```python
from concurrent.futures import as_completed

symbols = ['AAPL', 'MSFT', 'GOOGL', ...]

with ThreadPoolExecutor(max_workers=20) as executor:
    # Submit all tasks
    futures = {executor.submit(upload_file, symbol): symbol for symbol in symbols}

    # Process results as they complete (not in order!)
    for future in as_completed(futures):
        symbol = futures[future]  # Get original symbol
        try:
            result = future.result()
            print(f"✓ {symbol}: {result}")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
```

**Why use `as_completed()`?**
```
Without as_completed():
AAPL: [████████████] Done (12s)
MSFT: [████████] Done (8s)
GOOGL: [█████] Done (5s)
→ Wait for slowest (AAPL) before seeing any results

With as_completed():
GOOGL: [█████] Done (5s) ← Show result immediately!
MSFT: [████████] Done (8s) ← Show result immediately!
AAPL: [████████████] Done (12s) ← Show result immediately!
→ See progress as tasks finish
```

---

## 4. Pattern 1: Parallel File Uploads

### Step-by-Step: Uploading Thousands of Files

#### Step 1: Define the Upload Function

```python
import boto3
import logging

def upload_single_file(file_info):
    """
    Upload one file to S3.

    Args:
        file_info: Tuple of (local_path, s3_key)

    Returns:
        Tuple of (success: bool, s3_key: str)
    """
    local_path, s3_key = file_info
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(
            Filename=local_path,
            Bucket='us-equity-datalake',
            Key=s3_key
        )
        return True, s3_key
    except Exception as e:
        logging.error(f"Upload failed for {s3_key}: {e}")
        return False, s3_key
```

**Key points:**
- Takes one parameter (`file_info`) - simpler for parallel execution
- Returns tuple - easy to track success/failure
- Each thread gets its own `s3_client` - thread-safe

#### Step 2: Prepare the File List

```python
from pathlib import Path

def prepare_file_list(local_dir, s3_prefix):
    """
    Find all files to upload.

    Args:
        local_dir: Local directory (e.g., 'data/ticks/daily')
        s3_prefix: S3 prefix (e.g., 'data/ticks/daily')

    Returns:
        List of tuples: [(local_path, s3_key), ...]
    """
    local_dir = Path(local_dir)
    files = []

    for file_path in local_dir.rglob('*.parquet'):
        # Convert to relative path
        relative_path = file_path.relative_to(local_dir)

        # Build S3 key
        s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')

        files.append((str(file_path), s3_key))

    return files

# Example usage
files = prepare_file_list('data/ticks/daily', 'data/ticks/daily')
print(f"Found {len(files)} files to upload")
# Output: Found 75000 files to upload
```

**Why separate this step?**
- File discovery is sequential (can't parallelize)
- Do it once before parallel upload
- Clear separation of concerns

#### Step 3: Upload in Parallel

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_directory_parallel(files_to_upload, max_workers=20):
    """
    Upload files in parallel.

    Args:
        files_to_upload: List of (local_path, s3_key) tuples
        max_workers: Number of parallel uploads

    Returns:
        Dict with success/failure counts
    """
    total = len(files_to_upload)
    results = {'success': 0, 'failed': 0, 'errors': []}

    # Create executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_file = {
            executor.submit(upload_single_file, file_info): file_info
            for file_info in files_to_upload
        }

        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_file), 1):
            local_path, s3_key = future_to_file[future]

            try:
                success, key = future.result()
                if success:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(s3_key)
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(s3_key)
                logging.error(f"Unexpected error for {s3_key}: {e}")

            # Progress update
            if i % 100 == 0:  # Every 100 files
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

    return results
```

**How it works:**

```
1. Prepare task queue:
   future_to_file = {
       Future<upload AAPL/2024>: ('local/AAPL_2024.parq', 's3://...'),
       Future<upload MSFT/2024>: ('local/MSFT_2024.parq', 's3://...'),
       ...
   }

2. ThreadPoolExecutor distributes tasks:
   Thread 1: Upload AAPL/2024 → S3
   Thread 2: Upload MSFT/2024 → S3
   ...
   Thread 20: Upload TSLA/2024 → S3

3. as_completed() returns futures as they finish:
   First: GOOGL/2024 (5 seconds)
   Second: MSFT/2024 (8 seconds)
   Third: AAPL/2024 (12 seconds)
   ...

4. Collect results and track progress
```

#### Step 4: Put It All Together

```python
import logging

logging.basicConfig(level=logging.INFO)

# 1. Find files
files = prepare_file_list('data/ticks/daily', 'data/ticks/daily')
print(f"Found {len(files)} files")

# 2. Upload in parallel
results = upload_directory_parallel(files, max_workers=20)

# 3. Report results
print(f"\n✅ Upload complete:")
print(f"   Success: {results['success']}")
print(f"   Failed: {results['failed']}")
if results['errors']:
    print(f"   First 10 errors: {results['errors'][:10]}")
```

**Output:**
```
Found 75000 files
Progress: 100/75000 (0.1%)
Progress: 200/75000 (0.3%)
...
Progress: 75000/75000 (100.0%)

✅ Upload complete:
   Success: 74998
   Failed: 2
   First 10 errors: ['data/ticks/daily/XYZ/2024/ticks.parquet', ...]
```

---

## 5. Pattern 2: Parallel API Calls

### Step-by-Step: Fetching Data from APIs

#### Step 1: Define the API Call Function

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

def fetch_daily_ticks(symbol, date):
    """
    Fetch daily OHLCV data for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        date: Date to fetch (datetime.date object)

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    # Initialize Alpaca client (thread-safe)
    client = StockHistoricalDataClient(
        api_key=os.environ['ALPACA_API_KEY'],
        secret_key=os.environ['ALPACA_SECRET_KEY']
    )

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=date,
            end=date
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logging.warning(f"No data for {symbol} on {date}")
            return None

        return df

    except Exception as e:
        logging.error(f"API error for {symbol}: {e}")
        return None
```

**Thread safety notes:**
- Each thread creates its own `client` instance
- No shared state between threads
- API rate limits handled by Alpaca SDK

#### Step 2: Fetch Multiple Symbols in Parallel

```python
def fetch_all_symbols_parallel(symbols, date, max_workers=20):
    """
    Fetch daily ticks for all symbols in parallel.

    Args:
        symbols: List of stock symbols
        date: Date to fetch
        max_workers: Number of parallel API calls

    Returns:
        Dict mapping symbol → DataFrame
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all API calls
        future_to_symbol = {
            executor.submit(fetch_daily_ticks, symbol, date): symbol
            for symbol in symbols
        }

        # Collect results
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]

            try:
                df = future.result()
                if df is not None:
                    results[symbol] = df
                    print(f"✓ Fetched {symbol}")
                else:
                    print(f"✗ No data for {symbol}")
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")

    return results
```

**Performance comparison:**

```python
# Sequential: 5,000 symbols × 0.5s per API call = 2,500 seconds (~42 minutes)
for symbol in symbols:
    df = fetch_daily_ticks(symbol, date)

# Parallel (20 workers): 5,000 symbols / 20 workers × 0.5s = 125 seconds (~2 minutes)
fetch_all_symbols_parallel(symbols, date, max_workers=20)

# Result: 20x faster!
```

#### Step 3: Rate Limiting (Bonus)

Some APIs have rate limits. Here's how to handle them:

```python
import time
from threading import Lock

class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_calls_per_second=10):
        self.max_calls = max_calls_per_second
        self.calls = []
        self.lock = Lock()

    def wait_if_needed(self):
        """Wait if we've exceeded rate limit."""
        with self.lock:
            now = time.time()

            # Remove calls older than 1 second
            self.calls = [t for t in self.calls if now - t < 1.0]

            # Wait if at limit
            if len(self.calls) >= self.max_calls:
                sleep_time = 1.0 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.calls = []

            # Record this call
            self.calls.append(time.time())

# Usage
rate_limiter = RateLimiter(max_calls_per_second=10)

def fetch_with_rate_limit(symbol, date):
    """Fetch data with rate limiting."""
    rate_limiter.wait_if_needed()
    return fetch_daily_ticks(symbol, date)
```

**Why thread-safe?**
- `Lock()` ensures only one thread checks/updates `calls` at a time
- Prevents race conditions when multiple threads call API simultaneously

---

## 6. Pattern 3: Parallel Processing in Lambda

### Step-by-Step: Processing 5,000 Symbols in 15 Minutes

#### Step 1: Understanding Lambda Constraints

```python
# Lambda has:
# - Max execution time: 15 minutes (900 seconds)
# - Memory: 128 MB - 10,240 MB (we use 1024 MB)
# - CPU: Proportional to memory (1024 MB = ~0.6 vCPU)
# - /tmp storage: 512 MB - 10,240 MB

# For 5,000 symbols in 15 minutes:
# - Time per symbol: 900s / 5000 = 0.18 seconds
# - Must process ~28 symbols per second!
# - Requires parallel processing
```

#### Step 2: Determine Optimal Worker Count

```python
def lambda_handler(event, context):
    """Lambda handler with dynamic worker calculation."""

    # Calculate workers based on memory
    memory_mb = int(context.memory_limit_in_mb)
    max_workers = min(50, memory_mb // 128)
    # 1024 MB → 8 workers
    # 2048 MB → 16 workers
    # 3008 MB → 23 workers

    logging.info(f"Memory: {memory_mb} MB, Workers: {max_workers}")

    # Rest of logic...
```

**Why this formula?**
```
Each worker needs:
- Thread overhead: ~1 MB
- Boto3 S3 client: ~20 MB
- API client: ~30 MB
- Data buffers: ~50 MB
Total per worker: ~100 MB

Safe allocation: 128 MB per worker (leaves room for spikes)

Examples:
1024 MB / 128 MB = 8 workers  ✓ Good
2048 MB / 128 MB = 16 workers ✓ Better
3008 MB / 128 MB = 23 workers ✓ Best (but costs more)
```

#### Step 3: Process Symbols in Parallel

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def process_single_symbol(symbol, collection_date, s3_client):
    """
    Fetch data, save to /tmp, upload to S3.

    This runs in parallel across multiple threads.
    """
    try:
        # 1. Fetch from Alpaca API
        df = fetch_daily_ticks(symbol, collection_date)
        if df is None:
            return False, symbol, "No data"

        # 2. Save to /tmp
        tmp_path = f"/tmp/{symbol}_{collection_date}.parquet"
        df.to_parquet(tmp_path)

        # 3. Upload to S3
        year = collection_date.year
        s3_key = f"data/ticks/daily/{symbol}/{year}/ticks.parquet"
        s3_client.upload_file(
            Filename=tmp_path,
            Bucket='us-equity-datalake',
            Key=s3_key
        )

        # 4. Clean up /tmp
        os.remove(tmp_path)

        return True, symbol, None

    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        return False, symbol, str(e)

def lambda_handler(event, context):
    """Main Lambda handler with parallel processing."""

    # Load symbols
    symbols = load_stock_universe()  # 5,000 symbols
    collection_date = (datetime.now() - timedelta(days=1)).date()

    # Calculate workers
    memory_mb = int(context.memory_limit_in_mb)
    max_workers = min(50, memory_mb // 128)

    # Initialize S3 client (reused by all threads)
    s3_client = boto3.client('s3')

    results = {'success': 0, 'failed': 0, 'errors': []}

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_symbol, symbol, collection_date, s3_client): symbol
            for symbol in symbols
        }

        # Collect results
        for future in as_completed(futures):
            symbol = futures[future]

            try:
                success, sym, error = future.result()
                if success:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({'symbol': sym, 'error': error})

                # Check timeout (leave 1 minute buffer)
                remaining_ms = context.get_remaining_time_in_millis()
                if remaining_ms < 60000:
                    logging.warning(f"Approaching timeout, processed {results['success']} symbols")
                    break

            except Exception as e:
                logging.error(f"Unexpected error for {symbol}: {e}")
                results['failed'] += 1

    logging.info(f"Complete: {results['success']} success, {results['failed']} failed")

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

**Timeline:**

```
Time 0s:     Lambda cold starts (2-5s INIT phase)
Time 5s:     lambda_handler() starts
Time 5s:     Load 5,000 symbols, create 8 workers
Time 6s:     Start parallel processing

             Worker 1: AAPL → fetch → save → upload → done (3s)
             Worker 2: MSFT → fetch → save → upload → done (2.5s)
             Worker 3: GOOGL → fetch → save → upload → done (4s)
             ...
             Worker 8: TSLA → fetch → save → upload → done (3.5s)

             Workers automatically pick next symbol from queue

Time 630s:   All 5,000 symbols processed (8 workers × ~78 seconds each)
Time 630s:   Return results

Total: ~10.5 minutes (well within 15-minute limit)
```

---

## 7. Thread Safety and Shared State

### Understanding Thread Safety

**Thread-safe:** Multiple threads can access without causing problems.
**Not thread-safe:** Concurrent access causes data corruption or crashes.

#### Example 1: Thread-Safe (No Shared State)

```python
def upload_file(symbol):
    """Each thread has its own variables - SAFE."""

    # Local variables (separate for each thread)
    s3_client = boto3.client('s3')  # Each thread gets own client
    local_path = f"/tmp/{symbol}.parquet"  # Different file per thread

    # Upload
    s3_client.upload_file(local_path, 'bucket', f'{symbol}.parquet')
```

**Why safe?**
```
Thread 1: s3_client_1, local_path = "/tmp/AAPL.parquet"
Thread 2: s3_client_2, local_path = "/tmp/MSFT.parquet"
Thread 3: s3_client_3, local_path = "/tmp/GOOGL.parquet"

No shared data → No conflicts!
```

#### Example 2: NOT Thread-Safe (Shared Counter)

```python
# WRONG: Shared counter without lock
counter = 0  # Shared across all threads

def upload_file(symbol):
    global counter

    # Upload logic...

    counter += 1  # DANGER! Race condition
```

**Why not safe?**
```
Time 0ms: Thread 1 reads counter = 0
Time 1ms: Thread 2 reads counter = 0
Time 2ms: Thread 1 writes counter = 1
Time 3ms: Thread 2 writes counter = 1  ← Should be 2!

Result: Lost update! Counter = 1 instead of 2
```

#### Example 3: Thread-Safe (Using Lock)

```python
from threading import Lock

counter = 0
counter_lock = Lock()  # Protects counter

def upload_file(symbol):
    global counter

    # Upload logic...

    # Safely update counter
    with counter_lock:
        counter += 1  # Only one thread at a time
```

**Why safe now?**
```
Time 0ms: Thread 1 acquires lock, reads counter = 0
Time 1ms: Thread 2 tries to acquire lock → BLOCKED
Time 2ms: Thread 1 writes counter = 1, releases lock
Time 3ms: Thread 2 acquires lock, reads counter = 1
Time 4ms: Thread 2 writes counter = 2, releases lock

Result: Correct! Counter = 2
```

### Common Shared State Patterns

#### Pattern 1: Collecting Results (Thread-Safe)

```python
from concurrent.futures import ThreadPoolExecutor

def process_symbol(symbol):
    # Do work...
    return {'symbol': symbol, 'status': 'success'}

# Collect results - ThreadPoolExecutor handles thread safety
results = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_symbol, s) for s in symbols]

    # future.result() is thread-safe
    for future in as_completed(futures):
        result = future.result()
        results.append(result)  # Safe: only main thread appends
```

**Why safe?**
- Only one thread (main thread) calls `future.result()` and appends
- No concurrent writes to `results` list

#### Pattern 2: Progress Tracking (Needs Lock)

```python
from threading import Lock

progress = {'completed': 0, 'total': 5000}
progress_lock = Lock()

def process_symbol(symbol):
    # Do work...

    # Update progress
    with progress_lock:
        progress['completed'] += 1

        # Print every 100 symbols
        if progress['completed'] % 100 == 0:
            pct = progress['completed'] / progress['total'] * 100
            print(f"Progress: {progress['completed']}/{progress['total']} ({pct:.1f}%)")
```

**Why lock needed?**
- Multiple threads increment `completed` simultaneously
- Without lock: race condition (lost updates)

---

## 8. Error Handling in Parallel Operations

### Challenge: Errors in Parallel Code

```python
# Sequential: Easy to catch
for symbol in symbols:
    try:
        upload_file(symbol)
    except Exception as e:
        print(f"Error: {e}")  # Caught immediately

# Parallel: Errors happen in different threads!
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(upload_file, s) for s in symbols]
    # How do we catch errors?
```

### Solution 1: Catch in Future.result()

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_file(symbol):
    """May raise exception."""
    if symbol == 'BAD':
        raise ValueError("Invalid symbol!")
    # Upload logic...
    return f"Uploaded {symbol}"

symbols = ['AAPL', 'BAD', 'MSFT', 'GOOGL']

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(upload_file, s): s for s in symbols}

    for future in as_completed(futures):
        symbol = futures[future]

        try:
            result = future.result()  # This re-raises exception
            print(f"✓ {symbol}: {result}")
        except Exception as e:
            print(f"✗ {symbol}: {e}")  # Caught here!
```

**Output:**
```
✓ AAPL: Uploaded AAPL
✗ BAD: Invalid symbol!
✓ MSFT: Uploaded MSFT
✓ GOOGL: Uploaded GOOGL
```

### Solution 2: Catch Inside Worker Function

```python
def upload_file_safe(symbol):
    """Returns success/failure instead of raising."""
    try:
        # Upload logic...
        return True, symbol, None  # Success
    except Exception as e:
        return False, symbol, str(e)  # Failure

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(upload_file_safe, s) for s in symbols]

    for future in as_completed(futures):
        success, symbol, error = future.result()  # Never raises
        if success:
            print(f"✓ {symbol}")
        else:
            print(f"✗ {symbol}: {error}")
```

**When to use:**
- You want all tasks to complete (don't stop on first error)
- You need detailed error information
- You want to retry failed tasks

### Solution 3: Retry with Exponential Backoff

```python
import time
import random

def upload_with_retry(symbol, max_retries=3):
    """Retry failed uploads with exponential backoff."""

    for attempt in range(max_retries):
        try:
            # Upload logic...
            return True, symbol, None

        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                return False, symbol, str(e)

            # Wait before retry (exponential backoff + jitter)
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logging.warning(f"Retry {attempt + 1} for {symbol} after {wait_time:.2f}s")
            time.sleep(wait_time)

    return False, symbol, "Max retries exceeded"

# Use in ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(upload_with_retry, s) for s in symbols]
    # ...
```

**Backoff progression:**
```
Attempt 1: Immediate
Attempt 2: Wait 1-2 seconds (2^0 + jitter)
Attempt 3: Wait 2-3 seconds (2^1 + jitter)
Attempt 4: Wait 4-5 seconds (2^2 + jitter)
```

---

## 9. Complete Integration Example

### Putting It All Together: DailyTickCollector

This example shows how all the pieces integrate:

```python
"""
Complete parallel data collection system.

Components:
1. Parallel API calls (fetch data from Alpaca)
2. Parallel file I/O (save to /tmp)
3. Parallel S3 uploads
4. Thread-safe progress tracking
5. Error handling with retries
"""

import boto3
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe progress tracker
class ProgressTracker:
    """Track progress across multiple threads."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = Lock()

    def increment_success(self):
        with self.lock:
            self.completed += 1
            self._print_progress()

    def increment_failure(self):
        with self.lock:
            self.failed += 1
            self._print_progress()

    def _print_progress(self):
        if (self.completed + self.failed) % 100 == 0:
            total_done = self.completed + self.failed
            pct = total_done / self.total * 100
            logger.info(f"Progress: {total_done}/{self.total} ({pct:.1f}%) | "
                       f"✓ {self.completed} ✗ {self.failed}")

# Step 1: Worker function
def process_symbol(
    symbol: str,
    collection_date,
    s3_client,
    progress: ProgressTracker
) -> Tuple[bool, str, str]:
    """
    Fetch, save, and upload data for one symbol.

    Runs in parallel across multiple threads.
    """
    try:
        # 1. Fetch from API
        df = fetch_daily_ticks(symbol, collection_date)
        if df is None:
            progress.increment_failure()
            return False, symbol, "No data from API"

        # 2. Save to /tmp
        tmp_path = f"/tmp/{symbol}_{collection_date}.parquet"
        df.to_parquet(tmp_path)

        # 3. Upload to S3
        year = collection_date.year
        s3_key = f"data/ticks/daily/{symbol}/{year}/ticks.parquet"
        s3_client.upload_file(
            Filename=tmp_path,
            Bucket='us-equity-datalake',
            Key=s3_key
        )

        # 4. Clean up
        os.remove(tmp_path)

        progress.increment_success()
        return True, symbol, None

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        progress.increment_failure()
        return False, symbol, str(e)

# Step 2: Main orchestrator
class ParallelDataCollector:
    """Orchestrates parallel data collection."""

    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
        self.s3_client = boto3.client('s3')

    def collect_daily_ticks(
        self,
        symbols: List[str],
        collection_date = None
    ) -> Dict:
        """
        Collect daily ticks for all symbols in parallel.

        Args:
            symbols: List of stock symbols
            collection_date: Date to collect (default: yesterday)

        Returns:
            Dict with results and statistics
        """
        # Default to yesterday
        if collection_date is None:
            collection_date = (datetime.now() - timedelta(days=1)).date()

        logger.info(f"Starting collection for {len(symbols)} symbols on {collection_date}")
        logger.info(f"Using {self.max_workers} parallel workers")

        # Initialize progress tracker
        progress = ProgressTracker(total=len(symbols))

        # Results storage
        results = {
            'success': [],
            'failed': [],
            'errors': [],
            'statistics': {
                'total': len(symbols),
                'start_time': datetime.now()
            }
        }

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    process_symbol,
                    symbol,
                    collection_date,
                    self.s3_client,
                    progress
                ): symbol
                for symbol in symbols
            }

            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]

                try:
                    success, sym, error = future.result(timeout=60)  # 60s timeout per task

                    if success:
                        results['success'].append(sym)
                    else:
                        results['failed'].append(sym)
                        results['errors'].append({'symbol': sym, 'error': error})

                except TimeoutError:
                    logger.error(f"Timeout processing {symbol}")
                    results['failed'].append(symbol)
                    results['errors'].append({'symbol': symbol, 'error': 'Timeout'})
                    progress.increment_failure()

                except Exception as e:
                    logger.error(f"Unexpected error for {symbol}: {e}")
                    results['failed'].append(symbol)
                    results['errors'].append({'symbol': symbol, 'error': str(e)})
                    progress.increment_failure()

        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - results['statistics']['start_time']).total_seconds()

        results['statistics'].update({
            'end_time': end_time,
            'duration_seconds': duration,
            'success_count': len(results['success']),
            'failed_count': len(results['failed']),
            'success_rate': len(results['success']) / len(symbols) * 100
        })

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Collection Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total symbols: {len(symbols)}")
        logger.info(f"✓ Successful: {results['statistics']['success_count']}")
        logger.info(f"✗ Failed: {results['statistics']['failed_count']}")
        logger.info(f"Success rate: {results['statistics']['success_rate']:.2f}%")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Throughput: {len(symbols)/duration:.2f} symbols/second")
        logger.info(f"{'='*60}\n")

        return results

# Step 3: Usage
if __name__ == '__main__':
    # Example: Collect data for 100 symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', ...]  # 100 symbols

    collector = ParallelDataCollector(max_workers=20)
    results = collector.collect_daily_ticks(test_symbols)

    # Handle failures
    if results['failed']:
        logger.warning(f"\nFailed symbols ({len(results['failed'])}):")
        for error_info in results['errors'][:10]:  # Show first 10
            logger.warning(f"  {error_info['symbol']}: {error_info['error']}")
```

### How the Parts Fit Together

```
┌─────────────────────────────────────────────────────────────┐
│                   ParallelDataCollector                     │
│                                                             │
│  1. Initialize:                                             │
│     - Create ThreadPoolExecutor (20 workers)                │
│     - Create ProgressTracker (thread-safe counter)          │
│     - Create S3 client (shared across threads)              │
│                                                             │
│  2. Submit tasks:                                           │
│     - Loop through 5,000 symbols                            │
│     - Create future for each symbol                         │
│     - Store in future_to_symbol dict                        │
│                                                             │
│  3. Process in parallel:                                    │
│     ┌─────────────────────────────────────────────────┐    │
│     │  Worker Thread 1:                               │    │
│     │    process_symbol('AAPL')                       │    │
│     │      → fetch_daily_ticks()  (API call)          │    │
│     │      → save to /tmp                             │    │
│     │      → upload_file() to S3                      │    │
│     │      → progress.increment_success()             │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
│     ┌─────────────────────────────────────────────────┐    │
│     │  Worker Thread 2:                               │    │
│     │    process_symbol('MSFT')                       │    │
│     │      → ... (same steps)                         │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
│     ... (18 more worker threads)                            │
│                                                             │
│  4. Collect results:                                        │
│     - as_completed() returns futures as they finish         │
│     - Extract success/failure from each future              │
│     - Build results dictionary                              │
│                                                             │
│  5. Calculate statistics:                                   │
│     - Duration, throughput, success rate                    │
│     - Log summary                                           │
└─────────────────────────────────────────────────────────────┘
```

### Output Example

```
2024-12-20 11:00:00 - INFO - Starting collection for 5000 symbols on 2024-12-19
2024-12-20 11:00:00 - INFO - Using 20 parallel workers
2024-12-20 11:00:03 - INFO - Progress: 100/5000 (2.0%) | ✓ 98 ✗ 2
2024-12-20 11:00:06 - INFO - Progress: 200/5000 (4.0%) | ✓ 197 ✗ 3
...
2024-12-20 11:08:45 - INFO - Progress: 5000/5000 (100.0%) | ✓ 4985 ✗ 15
2024-12-20 11:08:45 - INFO -
============================================================
Collection Complete!
============================================================
Total symbols: 5000
✓ Successful: 4985
✗ Failed: 15
Success rate: 99.70%
Duration: 525.34 seconds
Throughput: 9.52 symbols/second
============================================================
```

---

## Summary

### Key Takeaways

**1. When to Use Threading:**
- ✅ S3 uploads/downloads
- ✅ API calls (Alpaca, SEC)
- ✅ Database queries
- ✅ Any I/O-bound task

**2. Core Tool: ThreadPoolExecutor**
```python
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(func, arg) for arg in args]
    for future in as_completed(futures):
        result = future.result()
```

**3. Performance Gains:**
| Operation | Sequential | Parallel (20 workers) | Speedup |
|-----------|------------|----------------------|---------|
| Upload 5,000 files | 5,000s (~1.4h) | 250s (~4min) | **20x** |
| Fetch 5,000 APIs | 2,500s (~42min) | 125s (~2min) | **20x** |
| Process in Lambda | 5,000s (timeout!) | 525s (~9min) | **9.5x** |

**4. Thread Safety Rules:**
- ✅ Each thread gets own variables (local scope)
- ✅ Shared resources need `Lock()`
- ✅ `as_completed()` handles result collection safely

**5. Error Handling:**
- Catch exceptions in `future.result()`
- Return success/failure tuples
- Implement retry with exponential backoff

### Next Steps

1. Review [LAMBDA_TUTORIAL.md](LAMBDA_TUTORIAL.md) for Lambda-specific parallel patterns
2. Review [S3_DATA_LAKE_TUTORIAL.md](S3_DATA_LAKE_TUTORIAL.md) for S3 upload patterns
3. Implement `ParallelDataCollector` class in your project
4. Start with small test (100 symbols) before scaling to 5,000
