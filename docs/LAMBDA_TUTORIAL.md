# AWS Lambda Tutorial for Data Lake Automation

## Table of Contents
1. [Introduction](#introduction)
2. [What is AWS Lambda?](#what-is-aws-lambda)
3. [Core Concepts](#core-concepts)
4. [Lambda Execution Lifecycle](#lambda-execution-lifecycle)
5. [Writing Lambda Handlers](#writing-lambda-handlers)
6. [Memory and Timeout Configuration](#memory-and-timeout-configuration)
7. [Temporary Storage (/tmp)](#temporary-storage-tmp)
8. [Environment Variables and Secrets](#environment-variables-and-secrets)
9. [IAM Roles and Permissions](#iam-roles-and-permissions)
10. [Integration with S3](#integration-with-s3)
11. [CloudWatch Logging](#cloudwatch-logging)
12. [Error Handling and Retries](#error-handling-and-retries)
13. [Cost Optimization](#cost-optimization)
14. [Production Best Practices](#production-best-practices)
15. [Complete Implementation Example](#complete-implementation-example)

---

## Introduction

This tutorial teaches you how to build AWS Lambda functions for automated data collection in the US Equity Data Lake. You'll learn how to:
- Write efficient Lambda handlers for stock data collection
- Configure memory, timeout, and concurrency settings
- Integrate with EventBridge (scheduled triggers) and S3 (data storage)
- Handle errors, retries, and edge cases
- Optimize costs while maintaining reliability

**What You'll Build:**
- `DailyTickCollector`: Collects OHLCV data daily at 6 AM ET
- `MinuteTickCollector`: Collects minute-level data during market hours
- `FundamentalCollector`: Fetches quarterly fundamental data from SEC
- `DataQualityValidator`: Validates data completeness and accuracy

---

## What is AWS Lambda?

**AWS Lambda** is a serverless compute service that runs your code in response to events without managing servers. You:
- Write code (Python, Node.js, Java, Go, etc.)
- Package code + dependencies
- Upload to Lambda
- Lambda handles infrastructure, scaling, availability

### Why Use Lambda for Data Lakes?

✅ **Serverless**: No servers to provision, patch, or manage
✅ **Auto-scaling**: Handles 1 or 10,000 concurrent executions automatically
✅ **Pay-per-use**: Only pay for compute time used (100ms granularity)
✅ **Integrated**: Native integration with S3, EventBridge, SQS, DynamoDB, etc.
✅ **Reliable**: Built-in retry logic and error handling
✅ **Fast**: Start executing code in milliseconds (warm starts)

### Lambda vs. EC2 for Data Collection

| Aspect             | Lambda                          | EC2                           |
|--------------------|---------------------------------|-------------------------------|
| **Management**     | Zero infrastructure management  | Manual provisioning/patching  |
| **Scaling**        | Automatic (0 → 1000s)           | Manual (ASG configuration)    |
| **Cost**           | Pay per execution               | Pay for uptime (24/7)         |
| **Cold start**     | 2-5 seconds                     | N/A (always running)          |
| **Max execution**  | 15 minutes                      | Unlimited                     |
| **Best for**       | Event-driven, short tasks       | Long-running, stateful tasks  |

**For daily data collection (5-15 min tasks):** Lambda is ideal.

---

## Core Concepts

### 1. Lambda Function

A **Lambda function** is your code + configuration:
- **Handler**: Entry point function (e.g., `lambda_handler()`)
- **Runtime**: Execution environment (Python 3.12, Node.js 20, etc.)
- **Memory**: 128 MB - 10,240 MB (CPU scales proportionally)
- **Timeout**: 1 second - 15 minutes
- **Environment variables**: Configuration passed to your code

### 2. Event Object

The **event** is the JSON input passed to your handler:

```python
# EventBridge scheduled event
{
  "version": "0",
  "id": "53dc4d37-cffa-4f76-80c9-8b7d4a4d2eaa",
  "detail-type": "Scheduled Event",
  "source": "aws.events",
  "time": "2024-12-19T11:00:00Z",
  "resources": ["arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule"],
  "detail": {
    "trigger": "scheduled",
    "collection_date": "auto",
    "symbols": "all"
  }
}
```

### 3. Context Object

The **context** provides runtime information:

```python
def lambda_handler(event, context):
    print(f"Request ID: {context.request_id}")
    print(f"Function name: {context.function_name}")
    print(f"Memory limit: {context.memory_limit_in_mb} MB")
    print(f"Time remaining: {context.get_remaining_time_in_millis()} ms")
```

**Common context attributes:**
- `request_id`: Unique ID for this invocation
- `function_name`: Name of Lambda function
- `memory_limit_in_mb`: Allocated memory
- `get_remaining_time_in_millis()`: Time left before timeout

### 4. Execution Environment

Lambda creates an **execution environment** for your function:
- Secure, isolated Linux container
- Runtime (Python interpreter, libraries)
- `/tmp` directory (512 MB - 10,240 MB storage)
- Environment variables
- Execution role (IAM permissions)

**Reuse:** Lambda may reuse the same environment for subsequent invocations (warm starts).

---

## Lambda Execution Lifecycle

### Complete Lifecycle: Cold Start → Warm Start → Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│ INVOCATION 1 (Cold Start)                                           │
├─────────────────────────────────────────────────────────────────────┤
│ 1. INIT Phase (runs ONCE per environment)                           │
│    ├─ Download code package from S3                                 │
│    ├─ Start runtime (Python interpreter)                            │
│    ├─ Load dependencies (boto3, pandas, etc.)                       │
│    ├─ Execute module-level code (imports, global variables)         │
│    └─ Duration: 2-5 seconds (or more with large dependencies)       │
│                                                                      │
│ 2. INVOKE Phase (runs EVERY time)                                   │
│    ├─ Execute lambda_handler(event, context)                        │
│    ├─ Your data collection logic runs here                          │
│    └─ Duration: depends on your code (e.g., 10-15 min)              │
│                                                                      │
│ 3. Return response                                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ INVOCATION 2 (Warm Start - minutes/hours later)                     │
├─────────────────────────────────────────────────────────────────────┤
│ 1. INIT Phase: SKIPPED (environment reused)                         │
│                                                                      │
│ 2. INVOKE Phase (runs EVERY time)                                   │
│    ├─ Execute lambda_handler(event, context)                        │
│    ├─ Global variables still in memory!                             │
│    └─ Duration: faster (no cold start overhead)                     │
│                                                                      │
│ 3. Return response                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Cold Start Breakdown

**Example: DailyTickCollector**

```python
# ===== INIT PHASE (cold start only) =====
import os
import json
import logging
from datetime import datetime, timedelta
import boto3
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient  # Takes time!

# Module-level initialization (runs ONCE per environment)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client (INIT phase)
s3_client = boto3.client('s3')

# Global variable (persists across invocations if environment reused)
execution_count = 0

# ===== INVOKE PHASE (every invocation) =====
def lambda_handler(event, context):
    global execution_count
    execution_count += 1

    logger.info(f"Execution #{execution_count} in this environment")
    logger.info(f"Request ID: {context.request_id}")

    # Extract input from event
    collection_date = event.get('detail', {}).get('collection_date', 'auto')
    symbols = event.get('detail', {}).get('symbols', 'all')

    # Collect data
    result = collect_daily_ticks(collection_date, symbols)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

**Cold Start Timeline:**
```
Time 0s:   EventBridge triggers Lambda
Time 0-2s: INIT phase (download code, start Python, import libraries)
Time 2s:   INVOKE phase starts, lambda_handler() executes
Time 12m:  lambda_handler() completes, response returned
```

**Warm Start Timeline:**
```
Time 0s:   EventBridge triggers Lambda (next day)
Time 0s:   INVOKE phase starts immediately (INIT skipped!)
Time 10m:  lambda_handler() completes (slightly faster)
```

### Optimizing Cold Starts

**Problem:** Large dependencies (pandas, numpy) cause 3-5 second cold starts.

**Solutions:**

1. **Use Lambda Layers** (shared dependencies):
```python
# Layer: common libraries (boto3, pandas, numpy)
# Function code: only your business logic
```

2. **Lazy imports** (import only when needed):
```python
def lambda_handler(event, context):
    # Only import pandas if we need it
    if event.get('use_pandas'):
        import pandas as pd
```

3. **Provisioned Concurrency** (pre-warmed environments):
```bash
aws lambda put-provisioned-concurrency-config \
  --function-name DailyTickCollector \
  --provisioned-concurrent-executions 1
```
**Cost:** ~$7/month per environment, but **zero cold starts**.

---

## Writing Lambda Handlers

### Basic Handler Structure

```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda handler for daily tick data collection.

    Args:
        event: EventBridge event with trigger details
        context: Lambda execution context

    Returns:
        dict: Response with statusCode and body
    """

    try:
        # 1. Log invocation details
        logger.info(f"Function triggered: {context.function_name}")
        logger.info(f"Request ID: {context.request_id}")
        logger.info(f"Event: {json.dumps(event)}")

        # 2. Extract input from event
        trigger_type = event.get('detail', {}).get('trigger', 'unknown')
        collection_date = event.get('detail', {}).get('collection_date', 'auto')

        # 3. Execute business logic
        result = process_data_collection(collection_date)

        # 4. Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data collection successful',
                'symbols_processed': result['count'],
                'execution_time': result['duration_seconds']
            })
        }

    except Exception as e:
        # 5. Handle errors
        logger.error(f"Error in lambda_handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'request_id': context.request_id
            })
        }

def process_data_collection(collection_date):
    """Business logic for data collection."""
    # Implementation here
    pass
```

### Handler for Daily Tick Collection

```python
import os
import json
import logging
from datetime import datetime, timedelta
import boto3
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients at module level (reused across invocations)
s3_client = boto3.client('s3')
alpaca_client = StockHistoricalDataClient(
    api_key=os.environ['ALPACA_API_KEY'],
    secret_key=os.environ['ALPACA_SECRET_KEY']
)

def lambda_handler(event, context):
    """Collect daily OHLCV data for all symbols."""

    logger.info(f"DailyTickCollector started - Request ID: {context.request_id}")

    # Determine collection date (yesterday's data)
    if event.get('detail', {}).get('collection_date') == 'auto':
        collection_date = (datetime.now() - timedelta(days=1)).date()
    else:
        collection_date = datetime.fromisoformat(
            event['detail']['collection_date']
        ).date()

    logger.info(f"Collecting data for date: {collection_date}")

    # Get stock universe from S3
    symbols = load_stock_universe()
    logger.info(f"Loaded {len(symbols)} symbols")

    # Collect data
    results = {
        'total_symbols': len(symbols),
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    for symbol in symbols:
        try:
            # Fetch daily OHLCV for this symbol
            ticks = fetch_daily_ticks(symbol, collection_date)

            if ticks:
                # Save to /tmp
                local_path = f"/tmp/{symbol}_{collection_date}.parquet"
                save_to_parquet(ticks, local_path)

                # Upload to S3
                year = collection_date.year
                s3_key = f"data/ticks/daily/{symbol}/{year}/ticks.parquet"
                upload_to_s3(local_path, s3_key)

                results['successful'] += 1

                # Clean up /tmp to avoid filling disk
                os.remove(local_path)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results['failed'] += 1
            results['errors'].append({'symbol': symbol, 'error': str(e)})

        # Check if approaching timeout (leave 30s buffer)
        if context.get_remaining_time_in_millis() < 30000:
            logger.warning("Approaching timeout, stopping early")
            break

    logger.info(f"Collection complete: {results['successful']} successful, "
                f"{results['failed']} failed")

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def load_stock_universe():
    """Load list of symbols from S3."""
    response = s3_client.get_object(
        Bucket=os.environ['S3_BUCKET'],
        Key='reference/stock_universe.json'
    )
    data = json.loads(response['Body'].read())
    return data['symbols']

def fetch_daily_ticks(symbol, date):
    """Fetch daily OHLCV for a symbol."""
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=date,
        end=date
    )
    bars = alpaca_client.get_stock_bars(request_params)
    return bars.df if not bars.df.empty else None

def save_to_parquet(dataframe, file_path):
    """Save DataFrame to Parquet file."""
    table = pa.Table.from_pandas(dataframe)
    pq.write_table(table, file_path, compression='snappy')

def upload_to_s3(local_path, s3_key):
    """Upload file to S3."""
    s3_client.upload_file(
        Filename=local_path,
        Bucket=os.environ['S3_BUCKET'],
        Key=s3_key
    )
    logger.info(f"Uploaded {s3_key}")
```

### Handler with Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def lambda_handler(event, context):
    """Collect data for multiple symbols in parallel."""

    symbols = load_stock_universe()
    collection_date = (datetime.now() - timedelta(days=1)).date()

    # Determine max workers based on Lambda memory
    # More memory = more vCPUs = more parallelism
    memory_mb = int(context.memory_limit_in_mb)
    max_workers = min(50, memory_mb // 128)  # 1 worker per 128 MB

    logger.info(f"Using {max_workers} parallel workers")

    results = {'successful': 0, 'failed': 0, 'errors': []}

    # Process symbols in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol, symbol, collection_date): symbol
            for symbol in symbols
        }

        # Process results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                future.result()
                results['successful'] += 1
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['failed'] += 1
                results['errors'].append({'symbol': symbol, 'error': str(e)})

    logger.info(f"Parallel processing complete: {results}")

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def process_symbol(symbol, date):
    """Process a single symbol (executed in parallel)."""
    ticks = fetch_daily_ticks(symbol, date)
    local_path = f"/tmp/{symbol}_{date}.parquet"
    save_to_parquet(ticks, local_path)

    s3_key = f"data/ticks/daily/{symbol}/{date.year}/ticks.parquet"
    upload_to_s3(local_path, s3_key)

    os.remove(local_path)
```

---

## Memory and Timeout Configuration

### Memory Configuration

Lambda memory ranges from **128 MB to 10,240 MB** (10 GB).

**Important:** CPU allocation scales with memory:
- 128 MB = ~0.08 vCPU
- 1,024 MB = 1 vCPU
- 1,792 MB = 1 full vCPU
- 10,240 MB = 6 vCPUs

**Rule of thumb:**
- **I/O-bound tasks** (API calls, S3 reads): 512-1024 MB
- **CPU-bound tasks** (data parsing, compression): 1792-3008 MB
- **Memory-intensive** (large DataFrames): 3008-10240 MB

### Choosing Memory for Data Collection

**DailyTickCollector (5,000 symbols, parallel processing):**

```python
# Test with different memory sizes
Memory: 512 MB  → Duration: 15 min → Cost: $0.25
Memory: 1024 MB → Duration: 8 min  → Cost: $0.21  ✅ BEST
Memory: 2048 MB → Duration: 5 min  → Cost: $0.26
```

**Why 1024 MB is optimal:**
- Enough parallelism (8-10 threads)
- Balances speed vs. cost
- Completes well within 15-min timeout

**Configure memory:**
```bash
aws lambda update-function-configuration \
  --function-name DailyTickCollector \
  --memory-size 1024
```

### Timeout Configuration

Lambda timeout ranges from **1 second to 15 minutes** (900 seconds).

**Choosing timeout:**
- Set timeout to **expected duration + 20% buffer**
- If your function typically runs 10 minutes, set timeout to 12 minutes

**Example:**
```bash
aws lambda update-function-configuration \
  --function-name DailyTickCollector \
  --timeout 720  # 12 minutes
```

**Handling timeouts in code:**
```python
def lambda_handler(event, context):
    symbols = load_stock_universe()

    for i, symbol in enumerate(symbols):
        # Check remaining time before processing next symbol
        remaining_ms = context.get_remaining_time_in_millis()

        if remaining_ms < 60000:  # Less than 1 minute left
            logger.warning(f"Approaching timeout at symbol {i}/{len(symbols)}")
            logger.warning(f"Processed {i} symbols, {len(symbols) - i} remaining")

            # Save progress and exit gracefully
            save_progress({'last_processed': i, 'total': len(symbols)})
            return {
                'statusCode': 206,  # Partial content
                'body': json.dumps({'status': 'partial', 'processed': i})
            }

        process_symbol(symbol)
```

### Ephemeral Storage Configuration (/tmp)

Lambda provides **512 MB to 10,240 MB** of `/tmp` storage (default: 512 MB).

**Use cases:**
- Downloading large files before uploading to S3
- Temporary Parquet files
- Caching data across invocations (if environment reused)

**Configure /tmp size:**
```bash
aws lambda update-function-configuration \
  --function-name DailyTickCollector \
  --ephemeral-storage '{"Size": 2048}'  # 2 GB
```

**Cost:** $0.0000000309/GB-second (very cheap)

**Best practice:**
```python
import os

def lambda_handler(event, context):
    # Process symbols in batches to avoid filling /tmp
    batch_size = 100

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]

        for symbol in batch:
            process_symbol(symbol)

        # Clean up /tmp after each batch
        cleanup_tmp_directory()

def cleanup_tmp_directory():
    """Remove all files in /tmp."""
    for filename in os.listdir('/tmp'):
        file_path = os.path.join('/tmp', filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
```

---

## Temporary Storage (/tmp)

### Characteristics

- **Default size:** 512 MB
- **Max size:** 10,240 MB (10 GB)
- **Lifecycle:** Persists during execution environment lifetime
- **Shared:** NOT shared across concurrent invocations
- **Cleaned:** May persist across invocations if environment reused

### Use Cases for /tmp

**1. Download → Process → Upload Pattern:**
```python
def process_large_file(s3_key):
    # Download from S3 to /tmp
    local_path = f"/tmp/{os.path.basename(s3_key)}"
    s3_client.download_file(BUCKET, s3_key, local_path)

    # Process file
    df = pd.read_csv(local_path)
    df_processed = transform_data(df)

    # Save processed data
    output_path = f"/tmp/processed_{os.path.basename(s3_key)}"
    df_processed.to_parquet(output_path)

    # Upload to S3
    s3_client.upload_file(output_path, BUCKET, f"processed/{s3_key}")

    # Clean up
    os.remove(local_path)
    os.remove(output_path)
```

**2. Caching Reference Data:**
```python
# Module-level cache
CACHE_FILE = '/tmp/stock_universe.json'

def load_stock_universe():
    """Load stock universe, use cache if available."""

    # Check if cache exists (from previous invocation)
    if os.path.exists(CACHE_FILE):
        logger.info("Using cached stock universe")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)

    # Download from S3
    logger.info("Downloading stock universe from S3")
    response = s3_client.get_object(Bucket=BUCKET, Key='reference/stock_universe.json')
    data = json.loads(response['Body'].read())

    # Save to cache for next invocation
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f)

    return data
```

### Monitoring /tmp Usage

```python
import shutil

def check_tmp_usage():
    """Check /tmp disk usage."""
    total, used, free = shutil.disk_usage('/tmp')

    logger.info(f"/tmp usage:")
    logger.info(f"  Total: {total / (1024**3):.2f} GB")
    logger.info(f"  Used:  {used / (1024**3):.2f} GB")
    logger.info(f"  Free:  {free / (1024**3):.2f} GB")

    if free < 100 * 1024 * 1024:  # Less than 100 MB free
        logger.warning("Low /tmp space, cleaning up...")
        cleanup_tmp_directory()
```

---

## Environment Variables and Secrets

### Environment Variables

Use environment variables for configuration:

```python
import os

# Access environment variables
S3_BUCKET = os.environ['S3_BUCKET']
ALPACA_API_KEY = os.environ['ALPACA_API_KEY']
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')  # Default to INFO
```

**Set environment variables:**
```bash
aws lambda update-function-configuration \
  --function-name DailyTickCollector \
  --environment "Variables={
    S3_BUCKET=us-equity-datalake,
    ALPACA_API_KEY=your-key,
    ALPACA_SECRET_KEY=your-secret,
    LOG_LEVEL=INFO
  }"
```

### Secrets Management (AWS Secrets Manager)

**Problem:** Hardcoding API keys in environment variables is insecure.

**Solution:** Use AWS Secrets Manager:

**1. Store secret:**
```bash
aws secretsmanager create-secret \
  --name prod/datalake/alpaca \
  --secret-string '{"api_key":"PK...","secret_key":"SK..."}'
```

**2. Retrieve in Lambda:**
```python
import boto3
import json

def get_alpaca_credentials():
    """Retrieve Alpaca API credentials from Secrets Manager."""

    secrets_client = boto3.client('secretsmanager')

    response = secrets_client.get_secret_value(
        SecretId='prod/datalake/alpaca'
    )

    secret = json.loads(response['SecretString'])
    return secret['api_key'], secret['secret_key']

# Initialize Alpaca client with secrets
api_key, secret_key = get_alpaca_credentials()
alpaca_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
```

**3. Grant Lambda permission:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:us-east-2:123456789012:secret:prod/datalake/alpaca-*"
    }
  ]
}
```

**Cost:** $0.40/secret/month + $0.05 per 10,000 API calls

### Caching Secrets (Reduce API Calls)

```python
import boto3
import json
from datetime import datetime, timedelta

# Module-level cache
_secret_cache = {}
_secret_expiry = {}

def get_secret(secret_id, ttl_minutes=60):
    """Get secret from Secrets Manager with caching."""

    now = datetime.now()

    # Check cache
    if secret_id in _secret_cache:
        if now < _secret_expiry[secret_id]:
            logger.info(f"Using cached secret: {secret_id}")
            return _secret_cache[secret_id]

    # Fetch from Secrets Manager
    logger.info(f"Fetching secret from Secrets Manager: {secret_id}")
    secrets_client = boto3.client('secretsmanager')
    response = secrets_client.get_secret_value(SecretId=secret_id)
    secret = json.loads(response['SecretString'])

    # Cache for TTL
    _secret_cache[secret_id] = secret
    _secret_expiry[secret_id] = now + timedelta(minutes=ttl_minutes)

    return secret
```

---

## IAM Roles and Permissions

### Lambda Execution Role

Every Lambda function needs an **execution role** (IAM role) that grants permissions.

**Minimum permissions:**
- Write logs to CloudWatch Logs
- (Optional) Read/write S3
- (Optional) Read Secrets Manager

### Creating Execution Role

**1. Trust policy (allows Lambda to assume this role):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**2. Permission policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-2:123456789012:log-group:/aws/lambda/DailyTickCollector:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::us-equity-datalake/data/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::us-equity-datalake"
    },
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:us-east-2:123456789012:secret:prod/datalake/*"
    }
  ]
}
```

**3. Create role via CLI:**
```bash
# Create role
aws iam create-role \
  --role-name LambdaDataLakeExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach managed policy for CloudWatch Logs
aws iam attach-role-policy \
  --role-name LambdaDataLakeExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Attach custom policy for S3 and Secrets Manager
aws iam put-role-policy \
  --role-name LambdaDataLakeExecutionRole \
  --policy-name DataLakeAccess \
  --policy-document file://permissions-policy.json
```

### Principle of Least Privilege

**Bad (overly permissive):**
```json
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*"
}
```

**Good (minimal permissions):**
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject"
  ],
  "Resource": "arn:aws:s3:::us-equity-datalake/data/ticks/daily/*"
}
```

---

## Integration with S3

### Reading from S3

```python
import boto3
import json

s3_client = boto3.client('s3')

def read_json_from_s3(bucket, key):
    """Read JSON file from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read())
    return data

def read_parquet_from_s3(bucket, key):
    """Read Parquet file from S3 into Pandas DataFrame."""
    import pandas as pd
    s3_path = f"s3://{bucket}/{key}"
    df = pd.read_parquet(s3_path)
    return df
```

### Writing to S3

```python
def write_json_to_s3(data, bucket, key):
    """Write JSON data to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data),
        ContentType='application/json'
    )

def write_parquet_to_s3(df, bucket, key):
    """Write DataFrame to S3 as Parquet."""
    import io
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Convert DataFrame to Parquet in memory
    table = pa.Table.from_pandas(df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression='snappy')

    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='application/octet-stream'
    )
```

### Streaming Large Files

**Problem:** Files > 1 GB don't fit in memory.

**Solution:** Stream data using boto3's upload_file:

```python
def download_and_upload_large_file(source_bucket, source_key, dest_bucket, dest_key):
    """Stream large file from one S3 bucket to another via /tmp."""

    # Download to /tmp
    local_path = f"/tmp/{os.path.basename(source_key)}"
    s3_client.download_file(source_bucket, source_key, local_path)

    # Process in chunks if needed
    process_file_in_chunks(local_path)

    # Upload to destination
    s3_client.upload_file(local_path, dest_bucket, dest_key)

    # Clean up
    os.remove(local_path)
```

---

## CloudWatch Logging

### Logging Basics

Lambda automatically sends logs to CloudWatch Logs:

```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("Processing started")
    logger.warning("Approaching timeout")
    logger.error("Failed to process symbol", exc_info=True)
```

**Log levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-critical issues)
- `ERROR`: Error messages (failures)
- `CRITICAL`: Critical errors (system-level failures)

### Structured Logging (JSON)

**Why:** Easier to parse, query, and analyze logs.

```python
import json
import logging

logger = logging.getLogger()

def log_json(level, message, **kwargs):
    """Log structured JSON."""
    log_entry = {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    logger.log(level, json.dumps(log_entry))

def lambda_handler(event, context):
    log_json(logging.INFO, 'Data collection started',
             symbols_count=5000,
             collection_date='2024-12-19')

    log_json(logging.ERROR, 'Failed to fetch data',
             symbol='AAPL',
             error='API timeout',
             retry_count=3)
```

**Output:**
```json
{"message": "Data collection started", "timestamp": "2024-12-19T11:00:00", "symbols_count": 5000, "collection_date": "2024-12-19"}
{"message": "Failed to fetch data", "timestamp": "2024-12-19T11:05:32", "symbol": "AAPL", "error": "API timeout", "retry_count": 3}
```

### Querying Logs with CloudWatch Insights

**Example queries:**

**1. Find errors:**
```
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 20
```

**2. Count symbols processed:**
```
fields @timestamp
| filter message = "Data collection started"
| stats count() as invocations, sum(symbols_count) as total_symbols
```

**3. Identify slow executions:**
```
fields @timestamp, @duration
| filter @duration > 600000  # More than 10 minutes
| sort @duration desc
```

---

## Error Handling and Retries

### Lambda Retry Behavior

**Synchronous invocations** (API Gateway, Lambda invoke):
- Lambda does NOT retry
- Caller is responsible for retries

**Asynchronous invocations** (EventBridge, S3 events, SNS):
- Lambda retries up to **2 times** (3 total attempts)
- Waits 1 minute, then 2 minutes between retries

### Handling Errors in Code

```python
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger()

def lambda_handler(event, context):
    try:
        result = collect_daily_ticks(event)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except ClientError as e:
        # AWS service error (S3, Secrets Manager, etc.)
        error_code = e.response['Error']['Code']
        logger.error(f"AWS service error: {error_code}", exc_info=True)

        if error_code in ['Throttling', 'TooManyRequestsException']:
            # Transient error - Lambda will retry
            raise  # Re-raise to trigger retry

        else:
            # Permanent error - don't retry
            return {
                'statusCode': 500,
                'body': json.dumps({'error': error_code})
            }

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise  # Re-raise to trigger retry
```

### Retry Logic in Code (Exponential Backoff)

```python
import time
import random

def fetch_with_retry(symbol, max_retries=3):
    """Fetch data with exponential backoff retry."""

    for attempt in range(max_retries):
        try:
            return fetch_daily_ticks(symbol)

        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                logger.error(f"Failed after {max_retries} attempts: {symbol}")
                raise

            # Calculate backoff delay (exponential + jitter)
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Retry {attempt + 1}/{max_retries} for {symbol} "
                          f"after {delay:.2f}s")
            time.sleep(delay)
```

### Dead-Letter Queue (DLQ)

**Purpose:** Capture events that fail after all retries.

**Setup:**
```python
# Create SQS DLQ
sqs_client = boto3.client('sqs')
dlq_response = sqs_client.create_queue(QueueName='LambdaDLQ')

# Configure Lambda DLQ
lambda_client = boto3.client('lambda')
lambda_client.update_function-configuration(
    FunctionName='DailyTickCollector',
    DeadLetterConfig={
        'TargetArn': dlq_response['QueueUrl']
    }
)
```

**Process DLQ messages:**
```python
def process_dlq_messages():
    """Investigate and replay failed Lambda invocations."""

    messages = sqs_client.receive_message(
        QueueUrl='https://sqs.us-east-2.amazonaws.com/123456789012/LambdaDLQ',
        MaxNumberOfMessages=10
    )

    for message in messages.get('Messages', []):
        # Parse failed event
        failed_event = json.loads(message['Body'])
        logger.info(f"Failed event: {failed_event}")

        # Investigate error, fix issue, then replay
        # ...

        # Delete from DLQ
        sqs_client.delete_message(
            QueueUrl='...',
            ReceiptHandle=message['ReceiptHandle']
        )
```

---

## Cost Optimization

### Lambda Pricing

**Components:**
1. **Requests:** $0.20 per 1 million requests
2. **Duration:** $0.0000166667 per GB-second
3. **Ephemeral storage:** $0.0000000309 per GB-second (beyond 512 MB)

**Example: DailyTickCollector**
- **Configuration:** 1024 MB, 10-minute execution, once daily
- **Requests:** 30/month
- **Duration:** 30 invocations × 600 seconds × 1 GB = 18,000 GB-seconds

**Cost calculation:**
```
Requests: 30 / 1,000,000 × $0.20 = $0.000006
Duration: 18,000 × $0.0000166667 = $0.30
Total: ~$0.30/month
```

**Compare to EC2 t3.small (24/7):**
- EC2 cost: ~$15/month
- **Lambda saves $14.70/month (98% savings!)**

### Optimization Strategies

**1. Right-size memory:**

Test different memory sizes to find optimal cost/performance:

```python
# Testing script
memory_sizes = [512, 1024, 1536, 2048, 3008]
results = []

for memory in memory_sizes:
    # Update Lambda memory
    update_function_configuration(memory)

    # Invoke and measure
    start = time.time()
    response = invoke_lambda()
    duration = time.time() - start

    # Calculate cost
    gb_seconds = (memory / 1024) * duration
    cost = gb_seconds * 0.0000166667

    results.append({
        'memory': memory,
        'duration': duration,
        'cost': cost
    })

# Find minimum cost
best = min(results, key=lambda x: x['cost'])
print(f"Optimal memory: {best['memory']} MB → ${best['cost']:.4f}")
```

**2. Use reserved concurrency (avoid throttling):**

```bash
# Reserve 5 concurrent executions for this function
aws lambda put-function-concurrency \
  --function-name DailyTickCollector \
  --reserved-concurrent-executions 5
```

**3. Use Lambda Layers (reduce package size):**

```bash
# Create layer with common dependencies
zip -r layer.zip python/
aws lambda publish-layer-version \
  --layer-name DataLakeCommonLibs \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.12
```

**4. Clean up old versions:**

Lambda keeps old versions by default. Delete unused versions:

```bash
# List versions
aws lambda list-versions-by-function --function-name DailyTickCollector

# Delete old version
aws lambda delete-function --function-name DailyTickCollector:1
```

---

## Production Best Practices

### 1. Use Lambda Layers for Shared Dependencies

**Benefits:**
- Reduce deployment package size
- Share libraries across functions
- Faster deployments

**Example:**
```bash
# Create layer with pandas, pyarrow, boto3
mkdir -p python/lib/python3.12/site-packages
pip install pandas pyarrow boto3 -t python/lib/python3.12/site-packages
zip -r layer.zip python/

# Publish layer
aws lambda publish-layer-version \
  --layer-name DataLakeLibs \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.12
```

### 2. Enable X-Ray Tracing

**Purpose:** Visualize execution flow and identify bottlenecks.

```bash
aws lambda update-function-configuration \
  --function-name DailyTickCollector \
  --tracing-config Mode=Active
```

**In code:**
```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch boto3, requests, etc.
patch_all()

@xray_recorder.capture('fetch_daily_ticks')
def fetch_daily_ticks(symbol, date):
    # X-Ray will trace this function
    ...
```

### 3. Set Up Alarms

```python
cloudwatch = boto3.client('cloudwatch')

# Alarm for errors
cloudwatch.put_metric_alarm(
    AlarmName='DailyTickCollectorErrors',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='Errors',
    Namespace='AWS/Lambda',
    Period=3600,
    Statistic='Sum',
    Threshold=1.0,
    Dimensions=[{'Name': 'FunctionName', 'Value': 'DailyTickCollector'}]
)

# Alarm for throttles
cloudwatch.put_metric_alarm(
    AlarmName='DailyTickCollectorThrottles',
    MetricName='Throttles',
    Namespace='AWS/Lambda',
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=1,
    Threshold=1.0,
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[{'Name': 'FunctionName', 'Value': 'DailyTickCollector'}]
)
```

### 4. Use Environment-Specific Configuration

```python
import os

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')

CONFIG = {
    'dev': {
        's3_bucket': 'us-equity-datalake-dev',
        'log_level': 'DEBUG',
        'symbols_limit': 100  # Test with fewer symbols
    },
    'prod': {
        's3_bucket': 'us-equity-datalake',
        'log_level': 'INFO',
        'symbols_limit': None  # Process all symbols
    }
}

config = CONFIG[ENVIRONMENT]
```

### 5. Implement Idempotency

**Why:** EventBridge may deliver the same event twice (rare).

```python
import hashlib
import json
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('LambdaExecutions')

def is_already_processed(event_id):
    """Check if event already processed."""
    response = table.get_item(Key={'event_id': event_id})
    return 'Item' in response

def mark_as_processed(event_id):
    """Mark event as processed."""
    table.put_item(Item={'event_id': event_id, 'timestamp': datetime.now().isoformat()})

def lambda_handler(event, context):
    # Generate deterministic event ID
    event_id = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()

    if is_already_processed(event_id):
        logger.info(f"Event {event_id} already processed, skipping")
        return {'statusCode': 200, 'body': 'Already processed'}

    # Process data
    result = collect_daily_ticks(event)

    # Mark as processed
    mark_as_processed(event_id)

    return result
```

---

## Complete Implementation Example

### Full DailyTickCollector Lambda Function

```python
"""
AWS Lambda function for daily stock tick data collection.
Triggered by EventBridge at 6 AM ET daily.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Initialize AWS clients (module level - reused across invocations)
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Configuration from environment variables
S3_BUCKET = os.environ['S3_BUCKET']
SECRET_ID = os.environ['ALPACA_SECRET_ID']
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'prod')

# Global variables (cached across warm starts)
_alpaca_client = None
_stock_universe = None


def lambda_handler(event, context):
    """
    Main Lambda handler for daily tick collection.

    Args:
        event: EventBridge scheduled event
        context: Lambda execution context

    Returns:
        dict: Response with status and statistics
    """

    logger.info(f"DailyTickCollector started - Request ID: {context.request_id}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Memory: {context.memory_limit_in_mb} MB")
    logger.info(f"Remaining time: {context.get_remaining_time_in_millis()} ms")

    start_time = datetime.now()

    try:
        # 1. Determine collection date
        collection_date = get_collection_date(event)
        logger.info(f"Collection date: {collection_date}")

        # 2. Load stock universe
        symbols = get_stock_universe()
        logger.info(f"Loaded {len(symbols)} symbols")

        # Apply limit for dev environment
        if ENVIRONMENT == 'dev':
            symbols = symbols[:100]
            logger.info(f"Dev mode: limited to {len(symbols)} symbols")

        # 3. Initialize Alpaca client
        alpaca = get_alpaca_client()

        # 4. Process symbols in parallel
        results = process_symbols_parallel(
            symbols,
            collection_date,
            alpaca,
            context
        )

        # 5. Log summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Collection complete: {results['successful']} successful, "
            f"{results['failed']} failed, {duration:.2f}s"
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data collection successful',
                'collection_date': str(collection_date),
                'total_symbols': len(symbols),
                'successful': results['successful'],
                'failed': results['failed'],
                'duration_seconds': duration,
                'environment': ENVIRONMENT
            })
        }

    except Exception as e:
        logger.error(f"Fatal error in lambda_handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'request_id': context.request_id
            })
        }


def get_collection_date(event):
    """Determine which date to collect data for."""
    detail = event.get('detail', {})

    if detail.get('collection_date') == 'auto':
        # Default: yesterday's data (market closes at 4 PM, we run at 6 AM next day)
        return (datetime.now() - timedelta(days=1)).date()
    elif detail.get('collection_date'):
        return datetime.fromisoformat(detail['collection_date']).date()
    else:
        return (datetime.now() - timedelta(days=1)).date()


def get_stock_universe():
    """Load stock universe from S3 with caching."""
    global _stock_universe

    # Return cached universe if available
    if _stock_universe is not None:
        logger.info("Using cached stock universe")
        return _stock_universe

    # Load from S3
    logger.info("Loading stock universe from S3")
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='reference/stock_universe.json'
        )
        data = json.loads(response['Body'].read())
        _stock_universe = data['symbols']
        return _stock_universe

    except ClientError as e:
        logger.error(f"Error loading stock universe: {e}")
        raise


def get_alpaca_client():
    """Initialize Alpaca client with credentials from Secrets Manager."""
    global _alpaca_client

    # Return cached client if available
    if _alpaca_client is not None:
        logger.info("Using cached Alpaca client")
        return _alpaca_client

    # Retrieve credentials from Secrets Manager
    logger.info(f"Retrieving Alpaca credentials from Secrets Manager: {SECRET_ID}")
    try:
        response = secrets_client.get_secret_value(SecretId=SECRET_ID)
        secret = json.loads(response['SecretString'])

        _alpaca_client = StockHistoricalDataClient(
            api_key=secret['api_key'],
            secret_key=secret['secret_key']
        )

        logger.info("Alpaca client initialized")
        return _alpaca_client

    except ClientError as e:
        logger.error(f"Error retrieving secrets: {e}")
        raise


def process_symbols_parallel(symbols, collection_date, alpaca_client, context):
    """Process multiple symbols in parallel using ThreadPoolExecutor."""

    # Determine max workers based on memory
    memory_mb = int(context.memory_limit_in_mb)
    max_workers = min(20, memory_mb // 128)
    logger.info(f"Using {max_workers} parallel workers")

    results = {
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(
                process_single_symbol,
                symbol,
                collection_date,
                alpaca_client
            ): symbol
            for symbol in symbols
        }

        # Process results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]

            try:
                future.result()
                results['successful'] += 1

                # Log progress every 100 symbols
                if results['successful'] % 100 == 0:
                    logger.info(f"Progress: {results['successful']}/{len(symbols)} symbols")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })

            # Check for timeout
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < 60000:  # Less than 1 minute
                logger.warning(
                    f"Approaching timeout with {remaining_ms}ms remaining. "
                    f"Processed {results['successful'] + results['failed']}/{len(symbols)}"
                )
                break

    return results


def process_single_symbol(symbol, collection_date, alpaca_client):
    """Fetch and save data for a single symbol."""

    # Fetch daily OHLCV
    bars_df = fetch_daily_bars(symbol, collection_date, alpaca_client)

    if bars_df is None or bars_df.empty:
        logger.warning(f"No data for {symbol} on {collection_date}")
        return

    # Save to /tmp
    local_path = f"/tmp/{symbol}_{collection_date}.parquet"
    save_to_parquet(bars_df, local_path)

    # Upload to S3
    year = collection_date.year
    s3_key = f"data/ticks/daily/{symbol}/{year}/ticks.parquet"
    upload_to_s3(local_path, s3_key)

    # Clean up /tmp
    os.remove(local_path)


def fetch_daily_bars(symbol, date, alpaca_client):
    """Fetch daily OHLCV bars from Alpaca."""

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=date,
            end=date
        )

        bars = alpaca_client.get_stock_bars(request)
        return bars.df

    except Exception as e:
        logger.error(f"Error fetching bars for {symbol}: {e}")
        raise


def save_to_parquet(df, file_path):
    """Save DataFrame to Parquet file."""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path, compression='snappy')


def upload_to_s3(local_path, s3_key):
    """Upload file to S3."""
    try:
        s3_client.upload_file(
            Filename=local_path,
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        logger.debug(f"Uploaded {s3_key}")

    except ClientError as e:
        logger.error(f"Error uploading {s3_key}: {e}")
        raise
```

### Deployment

```bash
# Package Lambda function
zip -r lambda_package.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
  --function-name DailyTickCollector \
  --runtime python3.12 \
  --role arn:aws:iam::123456789012:role/LambdaDataLakeExecutionRole \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_package.zip \
  --timeout 720 \
  --memory-size 1024 \
  --environment "Variables={
    S3_BUCKET=us-equity-datalake,
    ALPACA_SECRET_ID=prod/datalake/alpaca,
    LOG_LEVEL=INFO,
    ENVIRONMENT=prod
  }"
```

---

## Summary

**Key Takeaways:**

1. **Lambda** is serverless compute - no infrastructure to manage
2. **Cold starts** take 2-5 seconds (INIT phase), warm starts are instant
3. **Memory** determines CPU allocation (1,792 MB = 1 full vCPU)
4. **/tmp** provides 512 MB - 10 GB temporary storage
5. **Environment variables** for configuration, **Secrets Manager** for credentials
6. **IAM roles** grant permissions (S3, CloudWatch Logs, Secrets Manager)
7. **CloudWatch Logs** for monitoring, **X-Ray** for tracing
8. **Retries** automatic for async invocations (EventBridge, S3 events)
9. **Cost** is ~$0.30/month for daily data collection (vs. $15/month EC2)
10. **Best practices:** layers, logging, alarms, idempotency, timeout handling

**Next Steps:**
- Deploy Lambda functions for your data lake automation
- Set up EventBridge rules to trigger Lambda on schedule
- Configure CloudWatch alarms and DLQs
- Monitor costs and optimize memory/concurrency settings
