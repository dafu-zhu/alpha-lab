# AWS EventBridge Tutorial for Data Lake Automation

## Table of Contents
1. [Introduction](#introduction)
2. [What is AWS EventBridge?](#what-is-aws-eventbridge)
3. [Core Concepts](#core-concepts)
4. [EventBridge for Data Lake Automation](#eventbridge-for-data-lake-automation)
5. [Scheduled Rules with Cron Expressions](#scheduled-rules-with-cron-expressions)
6. [Setting Up EventBridge Rules](#setting-up-eventbridge-rules)
7. [Targeting Lambda Functions](#targeting-lambda-functions)
8. [Event Patterns vs. Scheduled Rules](#event-patterns-vs-scheduled-rules)
9. [Error Handling and Retries](#error-handling-and-retries)
10. [Monitoring and Debugging](#monitoring-and-debugging)
11. [Production Best Practices](#production-best-practices)
12. [Complete Implementation Example](#complete-implementation-example)

---

## Introduction

This tutorial teaches you how to use AWS EventBridge to automate data collection workflows for the US Equity Data Lake. You'll learn how to schedule Lambda functions to run at specific times (e.g., daily at 6 AM ET) to collect stock market data, fundamental data, and other time-sensitive information.

**What You'll Build:**
- Daily stock data collection triggered at 6 AM ET
- Automated fundamental data refresh (quarterly)
- Corporate actions monitoring (daily)
- Automated data quality checks (hourly)

---

## What is AWS EventBridge?

**AWS EventBridge** is a serverless event bus service that connects applications using events. Think of it as a smart scheduler and event router that can:

1. **Trigger actions on a schedule** (like cron jobs)
2. **React to AWS service events** (e.g., S3 object creation, EC2 state changes)
3. **Process custom application events**
4. **Route events to multiple targets** (Lambda, SQS, SNS, Step Functions, etc.)

### Why Use EventBridge for Data Lakes?

✅ **Serverless**: No infrastructure to manage
✅ **Reliable**: Built-in retry logic and dead-letter queues
✅ **Scalable**: Handles millions of events automatically
✅ **Cost-Effective**: Pay only for events processed ($1/million events)
✅ **Flexible**: Complex scheduling with cron expressions
✅ **Integrated**: Native integration with Lambda, S3, SNS, SQS, etc.

### EventBridge vs. CloudWatch Events

**Important:** AWS EventBridge is the evolution of CloudWatch Events. EventBridge has all CloudWatch Events capabilities plus:
- Support for third-party SaaS events
- Custom event buses
- Schema registry
- Archive and replay features

For new projects, **always use EventBridge**.

---

## Core Concepts

### 1. Events

An **event** is a JSON object describing a change in your environment:

```json
{
  "version": "0",
  "id": "53dc4d37-cffa-4f76-80c9-8b7d4a4d2eaa",
  "detail-type": "Scheduled Event",
  "source": "aws.events",
  "account": "123456789012",
  "time": "2024-12-19T11:00:00Z",
  "region": "us-east-2",
  "resources": [
    "arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule"
  ],
  "detail": {}
}
```

### 2. Event Buses

An **event bus** receives events. There are three types:
- **Default event bus**: Receives events from AWS services
- **Custom event bus**: For your application events
- **Partner event bus**: For third-party SaaS events

For data lake automation, you'll typically use the **default event bus**.

### 3. Rules

A **rule** matches incoming events and routes them to targets. Rules can be:
- **Scheduled rules**: Trigger on a schedule (cron/rate expressions)
- **Event pattern rules**: Trigger when events match a pattern

### 4. Targets

A **target** is what EventBridge invokes when an event matches a rule. Common targets:
- Lambda functions
- SQS queues
- SNS topics
- Step Functions state machines
- ECS tasks
- API Gateway endpoints

Each rule can have up to **5 targets**.

---

## EventBridge for Data Lake Automation

### Use Case: Daily Stock Data Collection

**Requirement:** Collect daily OHLCV data for 5,000 stocks every day at 6 AM ET (after market close at 4 PM ET the previous day).

**EventBridge Setup:**
1. Create a scheduled rule with cron expression: `cron(0 11 * * ? *)`
   - `11` = 11:00 UTC = 6:00 AM ET (UTC-5)
2. Target a Lambda function: `DailyTickCollector`
3. Lambda downloads yesterday's data and uploads to S3

**Architecture:**
```
EventBridge Rule                Lambda Function              S3 Data Lake
┌─────────────────┐            ┌──────────────────┐         ┌─────────────┐
│ DailyTickRule   │  triggers  │ DailyTickCollect │  writes │ ticks/daily/│
│ cron(0 11 * ..) │───────────>│ - Fetch OHLCV    │────────>│ AAPL/2024/  │
│                 │            │ - Parse data     │         │ ticks.parq  │
└─────────────────┘            │ - Upload S3      │         └─────────────┘
                               └──────────────────┘
```

---

## Scheduled Rules with Cron Expressions

EventBridge supports two types of schedules:

### 1. Rate Expressions (Simple)

**Format:** `rate(value unit)`

**Examples:**
```
rate(5 minutes)    → Every 5 minutes
rate(1 hour)       → Every hour
rate(1 day)        → Every day at the same time you created the rule
```

**Limitations:**
- Cannot specify exact time of day
- Good for simple intervals

### 2. Cron Expressions (Powerful)

**Format:** `cron(minute hour day-of-month month day-of-week year)`

**Fields:**
| Field         | Values          | Wildcards      |
|---------------|-----------------|----------------|
| Minute        | 0-59            | , - * /        |
| Hour          | 0-23 (UTC)      | , - * /        |
| Day-of-month  | 1-31            | , - * ? / L W  |
| Month         | 1-12 or JAN-DEC | , - * /        |
| Day-of-week   | 1-7 or SUN-SAT  | , - * ? L #    |
| Year          | 1970-2199       | , - * /        |

**Important:** EventBridge uses **UTC time zone** for cron expressions.

### Cron Examples for Data Lake

#### Daily at 6 AM ET (11 AM UTC)
```
cron(0 11 * * ? *)
```
- `0` = minute 0
- `11` = 11 AM UTC = 6 AM ET
- `*` = every day of month
- `*` = every month
- `?` = any day of week
- `*` = every year

**Use case:** Daily stock data collection after market close

#### Every 15 minutes during market hours (9:30 AM - 4 PM ET)
```
cron(0/15 14-20 ? * MON-FRI *)
```
- `0/15` = every 15 minutes starting at minute 0
- `14-20` = 9 AM - 3 PM UTC (covers 9:30 AM - 4 PM ET)
- `?` = any day of month
- `*` = every month
- `MON-FRI` = weekdays only
- `*` = every year

**Use case:** Intraday minute-level data collection

#### First day of every quarter at midnight ET
```
cron(0 5 1 1,4,7,10 ? *)
```
- `0` = minute 0
- `5` = 5 AM UTC = midnight ET
- `1` = first day of month
- `1,4,7,10` = January, April, July, October (Q1, Q2, Q3, Q4)
- `?` = any day of week
- `*` = every year

**Use case:** Quarterly fundamental data refresh

#### Every Sunday at 2 AM ET (maintenance window)
```
cron(0 7 ? * SUN *)
```
- `0` = minute 0
- `7` = 7 AM UTC = 2 AM ET
- `?` = any day of month
- `*` = every month
- `SUN` = Sunday only
- `*` = every year

**Use case:** Weekly data quality checks and cleanup

### Time Zone Conversion Table

| ET (US Eastern) | UTC  | Cron Hour |
|-----------------|------|-----------|
| 12 AM (midnight)| 5 AM | 5         |
| 6 AM            | 11 AM| 11        |
| 9:30 AM (open)  | 2:30 PM | 14    |
| 12 PM (noon)    | 5 PM | 17        |
| 4 PM (close)    | 9 PM | 21        |

**Note:** Adjust for daylight saving time (EDT = UTC-4, EST = UTC-5)

---

## Setting Up EventBridge Rules

### Method 1: AWS Console (Visual)

**Step 1: Create Rule**
1. Open AWS EventBridge console
2. Select **Rules** → **Create rule**
3. Enter rule name: `DailyTickCollectorRule`
4. Select event bus: **default**
5. Rule type: **Schedule**

**Step 2: Define Schedule**
1. Schedule pattern: **Cron-based schedule**
2. Cron expression: `cron(0 11 * * ? *)`
3. Time zone: **UTC**

**Step 3: Select Target**
1. Target type: **AWS service**
2. Target: **Lambda function**
3. Function: **DailyTickCollector**

**Step 4: Configure Target**
1. Configure input: **Constant (JSON text)**
2. JSON input:
```json
{
  "trigger": "scheduled",
  "collection_date": "auto",
  "symbols": "all"
}
```

**Step 5: Review and Create**

### Method 2: AWS CLI (Automation)

**Create Rule:**
```bash
aws events put-rule \
  --name DailyTickCollectorRule \
  --schedule-expression "cron(0 11 * * ? *)" \
  --description "Trigger daily stock data collection at 6 AM ET" \
  --state ENABLED
```

**Add Lambda Target:**
```bash
aws events put-targets \
  --rule DailyTickCollectorRule \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-2:123456789012:function:DailyTickCollector","Input"='{"trigger":"scheduled","collection_date":"auto"}'
```

**Grant EventBridge Permission to Invoke Lambda:**
```bash
aws lambda add-permission \
  --function-name DailyTickCollector \
  --statement-id AllowEventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule
```

### Method 3: Boto3 (Python SDK)

```python
import boto3
import json

# Initialize EventBridge client
events_client = boto3.client('events', region_name='us-east-2')

# Create scheduled rule
rule_response = events_client.put_rule(
    Name='DailyTickCollectorRule',
    ScheduleExpression='cron(0 11 * * ? *)',
    State='ENABLED',
    Description='Trigger daily stock data collection at 6 AM ET'
)

print(f"Rule ARN: {rule_response['RuleArn']}")

# Add Lambda function as target
target_response = events_client.put_targets(
    Rule='DailyTickCollectorRule',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:DailyTickCollector',
            'Input': json.dumps({
                'trigger': 'scheduled',
                'collection_date': 'auto',
                'symbols': 'all'
            })
        }
    ]
)

print(f"Target added: {target_response}")

# Grant EventBridge permission to invoke Lambda
lambda_client = boto3.client('lambda', region_name='us-east-2')
lambda_client.add_permission(
    FunctionName='DailyTickCollector',
    StatementId='AllowEventBridgeInvoke',
    Action='lambda:InvokeFunction',
    Principal='events.amazonaws.com',
    SourceArn=rule_response['RuleArn']
)

print("Permission granted to EventBridge")
```

---

## Targeting Lambda Functions

### Input Transformation

EventBridge can transform the event before sending to Lambda. This is useful for:
- Adding metadata
- Filtering fields
- Formatting data

**Example: Add collection date to event**

**Input Path:**
```json
{
  "time": "$.time"
}
```

**Input Template:**
```json
{
  "trigger": "scheduled",
  "event_time": "<time>",
  "collection_date": "auto",
  "symbols": "all",
  "environment": "production"
}
```

**Lambda receives:**
```json
{
  "trigger": "scheduled",
  "event_time": "2024-12-19T11:00:00Z",
  "collection_date": "auto",
  "symbols": "all",
  "environment": "production"
}
```

### Multiple Targets (Fan-Out Pattern)

**Use case:** Collect different data types in parallel

```python
# Create rule
events_client.put_rule(
    Name='DailyDataCollectionRule',
    ScheduleExpression='cron(0 11 * * ? *)',
    State='ENABLED'
)

# Add multiple Lambda targets
events_client.put_targets(
    Rule='DailyDataCollectionRule',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:DailyTickCollector',
            'Input': '{"data_type": "daily_ticks"}'
        },
        {
            'Id': '2',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:MinuteTickCollector',
            'Input': '{"data_type": "minute_ticks"}'
        },
        {
            'Id': '3',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:CorporateActionsCollector',
            'Input': '{"data_type": "corporate_actions"}'
        },
        {
            'Id': '4',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:FundamentalCollector',
            'Input': '{"data_type": "fundamentals"}'
        }
    ]
)
```

**Result:** All 4 Lambda functions execute **in parallel** at 6 AM ET.

---

## Event Patterns vs. Scheduled Rules

### Scheduled Rules (What We've Covered)

**Use when:** You need to run tasks at specific times or intervals

**Example:**
```json
{
  "ScheduleExpression": "cron(0 11 * * ? *)"
}
```

### Event Pattern Rules (Advanced)

**Use when:** You need to react to AWS service events

**Example: Trigger Lambda when new file uploaded to S3**

```json
{
  "source": ["aws.s3"],
  "detail-type": ["Object Created"],
  "detail": {
    "bucket": {
      "name": ["us-equity-datalake"]
    },
    "object": {
      "key": [{
        "prefix": "uploads/fundamentals/"
      }]
    }
  }
}
```

**Use case:** When a new fundamental data file is uploaded, trigger a Lambda function to parse and validate it.

**Example: Trigger when Lambda function fails**

```json
{
  "source": ["aws.lambda"],
  "detail-type": ["Lambda Function Execution State Change"],
  "detail": {
    "state": ["Failed"],
    "function-name": ["DailyTickCollector"]
  }
}
```

**Use case:** Send SNS alert when data collection fails.

---

## Error Handling and Retries

### EventBridge Retry Behavior

**Default Behavior:**
- EventBridge attempts to deliver each event to targets **up to 24 hours**
- Uses **exponential backoff** (delays increase: 0s, 1s, 2s, 4s, 8s, ...)
- Retries up to **185 times** for transient errors

**Transient Errors (will retry):**
- Lambda throttling
- Network timeouts
- Service unavailable (5xx errors)

**Permanent Errors (will NOT retry):**
- Lambda function not found
- Permission denied
- Invalid input

### Dead-Letter Queue (DLQ)

**Purpose:** Capture events that fail after all retries

**Setup:**
```python
# Create SQS queue for failed events
sqs_client = boto3.client('sqs')
dlq_response = sqs_client.create_queue(
    QueueName='EventBridgeDLQ',
    Attributes={
        'MessageRetentionPeriod': '1209600'  # 14 days
    }
)

dlq_arn = sqs_client.get_queue_attributes(
    QueueUrl=dlq_response['QueueUrl'],
    AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

# Configure DLQ on EventBridge target
events_client.put_targets(
    Rule='DailyTickCollectorRule',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-2:123456789012:function:DailyTickCollector',
            'DeadLetterConfig': {
                'Arn': dlq_arn
            },
            'RetryPolicy': {
                'MaximumRetryAttempts': 3,
                'MaximumEventAge': 3600  # 1 hour
            }
        }
    ]
)
```

**Benefits:**
- Investigate why events failed
- Replay failed events after fixing issues
- Set up alarms for DLQ message count

### Retry Policy Configuration

You can customize retry behavior:

```python
'RetryPolicy': {
    'MaximumRetryAttempts': 3,     # 0-185 (default: 185)
    'MaximumEventAge': 3600        # 60-86400 seconds (default: 86400)
}
```

**Recommendation for Data Lake:**
- `MaximumRetryAttempts`: 3 (fail fast for data collection)
- `MaximumEventAge`: 3600 (1 hour)
- **Reason:** If data collection fails, you want to know quickly and investigate, not retry for 24 hours

---

## Monitoring and Debugging

### CloudWatch Metrics

EventBridge publishes metrics to CloudWatch:

| Metric                 | Description                          |
|------------------------|--------------------------------------|
| `Invocations`          | Number of times targets were invoked |
| `FailedInvocations`    | Invocations that failed permanently  |
| `TriggeredRules`       | Rules that triggered                 |
| `ThrottledRules`       | Rules throttled due to rate limits   |

**View metrics:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Events \
  --metric-name Invocations \
  --dimensions Name=RuleName,Value=DailyTickCollectorRule \
  --start-time 2024-12-19T00:00:00Z \
  --end-time 2024-12-19T23:59:59Z \
  --period 3600 \
  --statistics Sum
```

### CloudWatch Logs

EventBridge does not log events by default. To enable logging:

1. **Create CloudWatch Log Group:**
```bash
aws logs create-log-group --log-group-name /aws/events/DailyTickCollectorRule
```

2. **Create EventBridge Archive (for replay):**
```python
events_client.create_archive(
    ArchiveName='DailyTickCollectorArchive',
    EventSourceArn='arn:aws:events:us-east-2:123456789012:event-bus/default',
    Description='Archive daily tick collector events for replay',
    RetentionDays=30,
    EventPattern=json.dumps({
        "source": ["aws.events"],
        "detail-type": ["Scheduled Event"],
        "resources": ["arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule"]
    })
)
```

### Testing Rules

**Manually trigger a rule (for testing):**
```bash
aws events put-events \
  --entries '[
    {
      "Source": "aws.events",
      "DetailType": "Scheduled Event",
      "Detail": "{\"trigger\":\"manual_test\"}",
      "Resources": ["arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule"]
    }
  ]'
```

**Check if rule is enabled:**
```bash
aws events describe-rule --name DailyTickCollectorRule
```

**List targets:**
```bash
aws events list-targets-by-rule --rule DailyTickCollectorRule
```

---

## Production Best Practices

### 1. Rule Naming Convention

Use descriptive, consistent names:
```
{Environment}{Frequency}{DataType}Collector
```

Examples:
- `ProdDailyTickCollector`
- `DevHourlyFundamentalCollector`
- `ProdWeeklyDataQualityCheck`

### 2. Use Input Constants for Configuration

Pass configuration via input JSON instead of hardcoding in Lambda:

```json
{
  "data_type": "daily_ticks",
  "start_date": "auto",
  "symbols": "all",
  "s3_bucket": "us-equity-datalake",
  "s3_prefix": "data/ticks/daily/",
  "alpaca_api_version": "v2",
  "retry_attempts": 3,
  "log_level": "INFO"
}
```

**Benefits:**
- Change configuration without modifying Lambda code
- Different configurations for different environments (dev/prod)

### 3. Implement Idempotency

**Problem:** EventBridge may deliver the same event multiple times (rare, but possible).

**Solution:** Make Lambda function idempotent:

```python
import hashlib
import json

def lambda_handler(event, context):
    # Generate unique execution ID from event
    event_id = event.get('id', 'unknown')
    execution_id = hashlib.sha256(json.dumps(event).encode()).hexdigest()

    # Check if already processed (use DynamoDB or S3)
    if is_already_processed(execution_id):
        logger.info(f"Event {event_id} already processed, skipping")
        return {'statusCode': 200, 'body': 'Already processed'}

    # Process data
    result = collect_daily_ticks(event)

    # Mark as processed
    mark_as_processed(execution_id)

    return result
```

### 4. Use Tags for Organization

Tag rules for cost tracking and organization:

```python
events_client.tag_resource(
    ResourceARN='arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule',
    Tags=[
        {'Key': 'Project', 'Value': 'USEquityDataLake'},
        {'Key': 'Environment', 'Value': 'Production'},
        {'Key': 'DataType', 'Value': 'DailyTicks'},
        {'Key': 'CostCenter', 'Value': 'DataEngineering'}
    ]
)
```

### 5. Set Up CloudWatch Alarms

Monitor for failures:

```python
cloudwatch_client = boto3.client('cloudwatch')

# Alarm for failed invocations
cloudwatch_client.put_metric_alarm(
    AlarmName='DailyTickCollectorFailures',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='FailedInvocations',
    Namespace='AWS/Events',
    Period=3600,
    Statistic='Sum',
    Threshold=1.0,
    ActionsEnabled=True,
    AlarmActions=[
        'arn:aws:sns:us-east-2:123456789012:DataLakeAlerts'
    ],
    AlarmDescription='Alert when daily tick collection fails',
    Dimensions=[
        {
            'Name': 'RuleName',
            'Value': 'DailyTickCollectorRule'
        }
    ]
)
```

### 6. Document Rule Purpose

Add description to every rule:

```python
events_client.put_rule(
    Name='DailyTickCollectorRule',
    ScheduleExpression='cron(0 11 * * ? *)',
    Description=(
        'Triggers daily stock data collection at 6 AM ET (11 AM UTC). '
        'Collects OHLCV data for all 5,000 symbols from previous trading day. '
        'Owner: data-engineering@company.com | '
        'Runbook: https://wiki.company.com/data-lake/daily-collection'
    ),
    State='ENABLED'
)
```

---

## Complete Implementation Example

### Scenario: Automated Data Lake Collection Pipeline

**Requirements:**
1. Daily OHLCV collection at 6 AM ET
2. Minute-level data collection every 15 minutes during market hours
3. Fundamental data refresh first Sunday of every month
4. Data quality check every hour
5. Error notifications via SNS

### Implementation

```python
import boto3
import json
from typing import Dict, List

class DataLakeEventBridge:
    """Manage EventBridge rules for US Equity Data Lake automation."""

    def __init__(self, region: str = 'us-east-2'):
        self.events_client = boto3.client('events', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

    def create_daily_tick_collection_rule(self):
        """Create rule for daily OHLCV data collection."""

        rule_name = 'ProdDailyTickCollector'
        lambda_function = 'DailyTickCollector'

        # Create rule
        rule_response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression='cron(0 11 * * ? *)',  # 6 AM ET
            State='ENABLED',
            Description='Collect daily OHLCV data at 6 AM ET after market close'
        )

        # Add Lambda target
        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{self.region}:{self.account_id}:function:{lambda_function}',
                    'Input': json.dumps({
                        'trigger': 'scheduled',
                        'data_type': 'daily_ticks',
                        'collection_date': 'auto',
                        'symbols': 'all',
                        's3_bucket': 'us-equity-datalake',
                        's3_prefix': 'data/ticks/daily/'
                    }),
                    'RetryPolicy': {
                        'MaximumRetryAttempts': 2,
                        'MaximumEventAge': 3600
                    },
                    'DeadLetterConfig': {
                        'Arn': f'arn:aws:sqs:{self.region}:{self.account_id}:EventBridgeDLQ'
                    }
                }
            ]
        )

        # Grant permission
        self._grant_lambda_permission(rule_name, lambda_function)

        print(f"Created rule: {rule_name}")
        return rule_response['RuleArn']

    def create_intraday_minute_collection_rule(self):
        """Create rule for minute-level data collection during market hours."""

        rule_name = 'ProdIntradayMinuteCollector'
        lambda_function = 'MinuteTickCollector'

        # Create rule (every 15 minutes, weekdays, 9 AM - 4 PM ET)
        rule_response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression='cron(0/15 14-21 ? * MON-FRI *)',
            State='ENABLED',
            Description='Collect minute-level data every 15 min during market hours'
        )

        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{self.region}:{self.account_id}:function:{lambda_function}',
                    'Input': json.dumps({
                        'trigger': 'scheduled',
                        'data_type': 'minute_ticks',
                        'collection_window': '15min',
                        's3_bucket': 'us-equity-datalake'
                    }),
                    'RetryPolicy': {
                        'MaximumRetryAttempts': 1,
                        'MaximumEventAge': 900  # 15 minutes
                    }
                }
            ]
        )

        self._grant_lambda_permission(rule_name, lambda_function)

        print(f"Created rule: {rule_name}")
        return rule_response['RuleArn']

    def create_monthly_fundamental_refresh_rule(self):
        """Create rule for monthly fundamental data refresh."""

        rule_name = 'ProdMonthlyFundamentalRefresh'
        lambda_function = 'FundamentalCollector'

        # First Sunday of every month at 2 AM ET
        rule_response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression='cron(0 7 ? * SUN#1 *)',
            State='ENABLED',
            Description='Refresh fundamental data first Sunday of month'
        )

        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{self.region}:{self.account_id}:function:{lambda_function}',
                    'Input': json.dumps({
                        'trigger': 'scheduled',
                        'data_type': 'fundamentals',
                        'refresh_type': 'full',
                        'source': 'sec_edgar'
                    })
                }
            ]
        )

        self._grant_lambda_permission(rule_name, lambda_function)

        print(f"Created rule: {rule_name}")
        return rule_response['RuleArn']

    def create_hourly_data_quality_check_rule(self):
        """Create rule for hourly data quality validation."""

        rule_name = 'ProdHourlyDataQualityCheck'
        lambda_function = 'DataQualityValidator'

        rule_response = self.events_client.put_rule(
            Name=rule_name,
            ScheduleExpression='rate(1 hour)',
            State='ENABLED',
            Description='Run data quality checks every hour'
        )

        self.events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': f'arn:aws:lambda:{self.region}:{self.account_id}:function:{lambda_function}',
                    'Input': json.dumps({
                        'trigger': 'scheduled',
                        'check_type': 'incremental',
                        'lookback_hours': 2
                    })
                }
            ]
        )

        self._grant_lambda_permission(rule_name, lambda_function)

        print(f"Created rule: {rule_name}")
        return rule_response['RuleArn']

    def _grant_lambda_permission(self, rule_name: str, function_name: str):
        """Grant EventBridge permission to invoke Lambda function."""

        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'AllowEventBridge{rule_name}',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f'arn:aws:events:{self.region}:{self.account_id}:rule/{rule_name}'
            )
        except self.lambda_client.exceptions.ResourceConflictException:
            # Permission already exists
            pass

    def setup_monitoring(self):
        """Set up CloudWatch alarms for EventBridge rules."""

        cloudwatch = boto3.client('cloudwatch', region_name=self.region)

        rules = [
            'ProdDailyTickCollector',
            'ProdIntradayMinuteCollector',
            'ProdMonthlyFundamentalRefresh'
        ]

        for rule_name in rules:
            cloudwatch.put_metric_alarm(
                AlarmName=f'{rule_name}-Failures',
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='FailedInvocations',
                Namespace='AWS/Events',
                Period=3600,
                Statistic='Sum',
                Threshold=1.0,
                ActionsEnabled=True,
                AlarmActions=[
                    f'arn:aws:sns:{self.region}:{self.account_id}:DataLakeAlerts'
                ],
                AlarmDescription=f'Alert when {rule_name} fails',
                Dimensions=[{'Name': 'RuleName', 'Value': rule_name}]
            )

        print("Monitoring alarms created")

    def deploy_all(self):
        """Deploy all EventBridge rules for data lake automation."""

        print("Deploying EventBridge rules for US Equity Data Lake...")

        self.create_daily_tick_collection_rule()
        self.create_intraday_minute_collection_rule()
        self.create_monthly_fundamental_refresh_rule()
        self.create_hourly_data_quality_check_rule()
        self.setup_monitoring()

        print("\n✅ All EventBridge rules deployed successfully!")

# Usage
if __name__ == '__main__':
    manager = DataLakeEventBridge(region='us-east-2')
    manager.deploy_all()
```

**Output:**
```
Deploying EventBridge rules for US Equity Data Lake...
Created rule: ProdDailyTickCollector
Created rule: ProdIntradayMinuteCollector
Created rule: ProdMonthlyFundamentalRefresh
Created rule: ProdHourlyDataQualityCheck
Monitoring alarms created

✅ All EventBridge rules deployed successfully!
```

---

## Summary

**Key Takeaways:**

1. **EventBridge** is a serverless event scheduler and router for automating data lake workflows
2. **Scheduled rules** use cron expressions to trigger Lambda at specific times (use UTC)
3. **Targets** can be Lambda functions, SQS queues, SNS topics, Step Functions, etc.
4. **Retry logic** is built-in with exponential backoff (up to 24 hours by default)
5. **Dead-letter queues** capture events that fail after all retries
6. **Monitoring** via CloudWatch metrics and alarms is essential for production
7. **Best practices:** idempotency, input constants, tagging, documentation

**Next Steps:**
- Read the Lambda tutorial to learn how to write Lambda functions triggered by EventBridge
- Set up EventBridge rules for your data lake automation
- Configure CloudWatch alarms and DLQs for production reliability
