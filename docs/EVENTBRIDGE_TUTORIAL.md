# AWS EventBridge Tutorial for Data Lake Automation

## Introduction

This tutorial teaches you how to use AWS EventBridge to automate data collection for the US Equity Data Lake. We focus **only** on scheduled rules (cron expressions) - the core feature you'll actually use.

### What You'll Build

- Daily stock data collection triggered at 6 AM ET
- Automated fundamental data refresh (monthly)
- Corporate actions monitoring (daily)

### Prerequisites

- AWS CLI configured
- Python 3.8+ with boto3
- Basic understanding of cron expressions

---

## Table of Contents

1. [Core EventBridge Methods](#1-core-eventbridge-methods)
2. [Cron Expressions for Data Collection](#2-cron-expressions-for-data-collection)
3. [Creating Scheduled Rules](#3-creating-scheduled-rules)
4. [Targeting Lambda Functions](#4-targeting-lambda-functions)
5. [Error Handling and Retries](#5-error-handling-and-retries)
6. [Complete Implementation](#6-complete-implementation)

---

## 1. Core EventBridge Methods

### Methods Used in Data Lake Project

```python
import boto3

events_client = boto3.client('events', region_name='us-east-2')

# Core methods we use:
events_client.put_rule(Name, ScheduleExpression, State, Description)
events_client.put_targets(Rule, Targets)
lambda_client.add_permission(FunctionName, StatementId, Action, Principal, SourceArn)
```

### Why These Methods?

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `put_rule()` | Create/update scheduled rule | Define when data collection runs (e.g., daily at 6 AM ET) |
| `put_targets()` | Attach Lambda function to rule | Tell EventBridge which Lambda to invoke |
| `add_permission()` | Grant EventBridge permission | Allow EventBridge to invoke your Lambda function |

**That's it!** Unlike S3, EventBridge has a minimal API surface for our use case.

---

## 2. Cron Expressions for Data Collection

### Cron Expression Format

```
cron(minute hour day-of-month month day-of-week year)
```

**CRITICAL:** EventBridge uses **UTC time zone**. You must convert ET to UTC.

| ET (US Eastern) | UTC  | Cron Hour |
|-----------------|------|-----------|
| 12 AM (midnight)| 5 AM | 5         |
| 6 AM            | 11 AM| 11        |
| 9:30 AM (open)  | 2:30 PM | 14    |
| 4 PM (close)    | 9 PM | 21        |

**Note:** ET = UTC-5 (winter), EDT = UTC-4 (summer). For simplicity, we use UTC-5.

### Cron Expressions for Data Lake

#### Daily at 6 AM ET (11 AM UTC)
```python
'cron(0 11 * * ? *)'
```
- `0` = minute 0
- `11` = 11 AM UTC = 6 AM ET
- `*` = every day of month
- `*` = every month
- `?` = any day of week
- `*` = every year

**Use case:** Daily stock data collection after market close

#### First Sunday of every month at 2 AM ET (7 AM UTC)
```python
'cron(0 7 ? * SUN#1 *)'
```
- `0` = minute 0
- `7` = 7 AM UTC = 2 AM ET
- `?` = any day of month
- `*` = every month
- `SUN#1` = first Sunday
- `*` = every year

**Use case:** Monthly fundamental data refresh

#### Every weekday at 6 AM ET
```python
'cron(0 11 ? * MON-FRI *)'
```
- `0` = minute 0
- `11` = 11 AM UTC = 6 AM ET
- `?` = any day of month
- `*` = every month
- `MON-FRI` = weekdays only
- `*` = every year

**Use case:** Daily tick collection (skip weekends)

---

## 3. Creating Scheduled Rules

### put_rule() Method

**Purpose:** Create or update a scheduled rule.

**All Parameters Explained:**

```python
response = events_client.put_rule(
    Name='DailyTickCollectorRule',              # REQUIRED: Rule name (unique per account/region)
    ScheduleExpression='cron(0 11 * * ? *)',    # REQUIRED: Cron or rate expression
    State='ENABLED',                            # REQUIRED: 'ENABLED' or 'DISABLED'
    Description='Trigger daily at 6 AM ET',     # OPTIONAL: Human-readable description
    EventBusName='default',                     # OPTIONAL: Event bus (default: 'default')
    RoleArn='...'                               # OPTIONAL: IAM role (not needed for Lambda targets)
)

# Response:
{
    'RuleArn': 'arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule'
}
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Name` | str | ✅ Yes | Rule name. Use pattern: `{Frequency}{DataType}Collector` (e.g., `DailyTickCollector`) |
| `ScheduleExpression` | str | ✅ Yes | Cron expression defining when to trigger (see section 2) |
| `State` | str | ✅ Yes | `'ENABLED'` to activate, `'DISABLED'` to pause without deleting |
| `Description` | str | ⚠️ Recommended | Document what this rule does, owner, runbook link |
| `EventBusName` | str | ❌ Optional | Always use `'default'` for scheduled rules (auto default) |
| `RoleArn` | str | ❌ Not used | Only needed for cross-account targets (we don't use this) |

**Example Usage:**

```python
import boto3
import logging

events_client = boto3.client('events', region_name='us-east-2')

def create_daily_tick_collection_rule():
    """
    Create rule to trigger daily tick collection at 6 AM ET.

    Runs: Monday-Friday at 11 AM UTC (6 AM ET)
    Targets: DailyTickCollector Lambda function
    """
    try:
        response = events_client.put_rule(
            Name='DailyTickCollectorRule',
            ScheduleExpression='cron(0 11 ? * MON-FRI *)',  # Weekdays at 6 AM ET
            State='ENABLED',
            Description=(
                'Trigger daily stock data collection at 6 AM ET (11 AM UTC). '
                'Collects OHLCV data for all 5,000 symbols from previous trading day. '
                'Owner: data-engineering@company.com'
            )
        )

        rule_arn = response['RuleArn']
        logging.info(f"Created rule: {rule_arn}")
        return rule_arn

    except Exception as e:
        logging.error(f"Failed to create rule: {e}")
        raise

# Usage
rule_arn = create_daily_tick_collection_rule()
```

---

## 4. Targeting Lambda Functions

### put_targets() Method

**Purpose:** Attach one or more Lambda functions to a rule.

**All Parameters Explained:**

```python
response = events_client.put_targets(
    Rule='DailyTickCollectorRule',              # REQUIRED: Rule name
    Targets=[                                   # REQUIRED: List of targets (max 5)
        {
            'Id': '1',                          # REQUIRED: Unique ID within this rule
            'Arn': 'arn:aws:lambda:...',        # REQUIRED: Lambda function ARN
            'Input': '{"key":"value"}',         # OPTIONAL: JSON input to Lambda
            'RetryPolicy': {...},               # OPTIONAL: Retry configuration
            'DeadLetterConfig': {...}           # OPTIONAL: DLQ for failed events
        }
    ]
)

# Response:
{
    'FailedEntryCount': 0,
    'FailedEntries': []
}
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Rule` | str | ✅ Yes | Rule name created in `put_rule()` |
| `Targets` | list | ✅ Yes | List of Lambda functions to invoke (we typically use 1 target per rule) |
| `Targets[].Id` | str | ✅ Yes | Unique ID (use `'1'`, `'2'`, etc.) |
| `Targets[].Arn` | str | ✅ Yes | Lambda function ARN (format: `arn:aws:lambda:{region}:{account}:function:{name}`) |
| `Targets[].Input` | str | ⚠️ Recommended | JSON string passed to Lambda event (config, parameters) |
| `Targets[].RetryPolicy` | dict | ⚠️ Recommended | Controls retry behavior (see section 5) |
| `Targets[].DeadLetterConfig` | dict | ⚠️ Recommended | SQS queue for failed events (see section 5) |

**Example Usage:**

```python
import json

def attach_lambda_target(rule_name, lambda_function_arn):
    """
    Attach Lambda function to EventBridge rule.

    Args:
        rule_name: EventBridge rule name
        lambda_function_arn: Lambda function ARN
    """
    try:
        response = events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': lambda_function_arn,
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
                        'MaximumEventAge': 3600  # 1 hour
                    }
                }
            ]
        )

        if response['FailedEntryCount'] == 0:
            logging.info(f"Attached Lambda to rule: {rule_name}")
        else:
            logging.error(f"Failed to attach target: {response['FailedEntries']}")
            raise Exception(f"Failed to attach target")

    except Exception as e:
        logging.error(f"Error attaching target: {e}")
        raise

# Usage
attach_lambda_target(
    rule_name='DailyTickCollectorRule',
    lambda_function_arn='arn:aws:lambda:us-east-2:123456789012:function:DailyTickCollector'
)
```

### add_permission() - Grant EventBridge Permission

**Purpose:** Allow EventBridge to invoke your Lambda function.

**All Parameters Explained:**

```python
lambda_client = boto3.client('lambda', region_name='us-east-2')

response = lambda_client.add_permission(
    FunctionName='DailyTickCollector',          # REQUIRED: Lambda function name or ARN
    StatementId='AllowEventBridgeInvoke',       # REQUIRED: Unique statement ID
    Action='lambda:InvokeFunction',             # REQUIRED: Permission to grant (always this)
    Principal='events.amazonaws.com',           # REQUIRED: Who can invoke (always EventBridge)
    SourceArn='arn:aws:events:...'              # REQUIRED: Specific rule ARN (security)
)
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `FunctionName` | str | ✅ Yes | Lambda function name (e.g., `'DailyTickCollector'`) |
| `StatementId` | str | ✅ Yes | Unique ID. Use pattern: `AllowEventBridge{RuleName}` |
| `Action` | str | ✅ Yes | Always `'lambda:InvokeFunction'` |
| `Principal` | str | ✅ Yes | Always `'events.amazonaws.com'` |
| `SourceArn` | str | ✅ Yes | Rule ARN from `put_rule()` (ensures only this rule can invoke) |

**Example Usage:**

```python
def grant_eventbridge_permission(function_name, rule_arn, rule_name):
    """
    Grant EventBridge permission to invoke Lambda function.

    Args:
        function_name: Lambda function name
        rule_arn: EventBridge rule ARN
        rule_name: EventBridge rule name (for statement ID)
    """
    lambda_client = boto3.client('lambda', region_name='us-east-2')

    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'AllowEventBridge{rule_name}',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=rule_arn
        )
        logging.info(f"Granted permission for {rule_name} to invoke {function_name}")

    except lambda_client.exceptions.ResourceConflictException:
        # Permission already exists - this is fine
        logging.info(f"Permission already exists for {function_name}")

    except Exception as e:
        logging.error(f"Error granting permission: {e}")
        raise

# Usage
grant_eventbridge_permission(
    function_name='DailyTickCollector',
    rule_arn='arn:aws:events:us-east-2:123456789012:rule/DailyTickCollectorRule',
    rule_name='DailyTickCollectorRule'
)
```

---

## 5. Error Handling and Retries

### RetryPolicy Configuration

**Purpose:** Control how EventBridge retries failed Lambda invocations.

**All Parameters Explained:**

```python
'RetryPolicy': {
    'MaximumRetryAttempts': 2,      # 0-185 (default: 185)
    'MaximumEventAge': 3600         # 60-86400 seconds (default: 86400)
}
```

**Parameter Details:**

| Parameter | Default | Our Setting | Why? |
|-----------|---------|-------------|------|
| `MaximumRetryAttempts` | 185 | **2** | Retry twice (3 total attempts). Fail fast for data collection. |
| `MaximumEventAge` | 86400 (24h) | **3600 (1h)** | If collection fails, we want to know within 1 hour, not wait 24 hours. |

**EventBridge Retry Behavior:**

```
Attempt 1 (immediate): Lambda fails
   ↓
Wait 1 minute
   ↓
Attempt 2 (retry 1): Lambda fails
   ↓
Wait 2 minutes
   ↓
Attempt 3 (retry 2): Lambda fails
   ↓
Send to Dead-Letter Queue (if configured)
```

**When EventBridge Retries:**
- ✅ Lambda throttled (concurrent execution limit)
- ✅ Lambda timeout
- ✅ Lambda internal error (unhandled exception)
- ❌ Lambda returns error response (we handle this, no retry)

### DeadLetterConfig - Failed Event Handling

**Purpose:** Capture events that fail after all retries.

**All Parameters Explained:**

```python
'DeadLetterConfig': {
    'Arn': 'arn:aws:sqs:us-east-2:123456789012:EventBridgeDLQ'
}
```

**Parameter Details:**

| Parameter | Type | Required | Purpose in Data Lake |
|-----------|------|----------|---------------------|
| `Arn` | str | ✅ Yes | SQS queue ARN to send failed events |

**Why Use DLQ:**
- Investigate why data collection failed
- Manually replay failed collections
- Set up CloudWatch alarms on DLQ message count

**Example Setup:**

```python
def create_dlq_and_attach():
    """
    Create SQS Dead-Letter Queue and attach to EventBridge target.

    Returns:
        DLQ ARN
    """
    sqs_client = boto3.client('sqs', region_name='us-east-2')

    # Create SQS queue
    queue_response = sqs_client.create_queue(
        QueueName='EventBridgeDLQ',
        Attributes={
            'MessageRetentionPeriod': '1209600'  # 14 days
        }
    )

    # Get queue ARN
    queue_url = queue_response['QueueUrl']
    attrs = sqs_client.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['QueueArn']
    )
    dlq_arn = attrs['Attributes']['QueueArn']

    logging.info(f"Created DLQ: {dlq_arn}")
    return dlq_arn

# Usage in put_targets
dlq_arn = create_dlq_and_attach()

events_client.put_targets(
    Rule='DailyTickCollectorRule',
    Targets=[
        {
            'Id': '1',
            'Arn': lambda_function_arn,
            'RetryPolicy': {
                'MaximumRetryAttempts': 2,
                'MaximumEventAge': 3600
            },
            'DeadLetterConfig': {
                'Arn': dlq_arn  # Send failed events here
            }
        }
    ]
)
```

---

## 6. Complete Implementation

### Production-Ready EventBridge Setup

This class combines all methods into a production-ready implementation:

```python
"""
US Equity Data Lake EventBridge Automation

Sets up scheduled rules for:
- Daily tick collection (6 AM ET, weekdays)
- Monthly fundamental refresh (first Sunday, 2 AM ET)
"""

import boto3
import json
import logging
from typing import Dict


class DataLakeEventBridge:
    """Manage EventBridge rules for US Equity Data Lake automation."""

    def __init__(self, region: str = 'us-east-2'):
        self.region = region
        self.events_client = boto3.client('events', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sqs_client = boto3.client('sqs', region_name=region)

        # Get AWS account ID
        sts_client = boto3.client('sts')
        self.account_id = sts_client.get_caller_identity()['Account']

        self.logger = logging.getLogger(__name__)

    def create_rule(
        self,
        name: str,
        schedule: str,
        description: str
    ) -> str:
        """
        Create EventBridge scheduled rule.

        Args:
            name: Rule name
            schedule: Cron expression
            description: Human-readable description

        Returns:
            Rule ARN
        """
        try:
            response = self.events_client.put_rule(
                Name=name,
                ScheduleExpression=schedule,
                State='ENABLED',
                Description=description
            )

            rule_arn = response['RuleArn']
            self.logger.info(f"Created rule: {name}")
            return rule_arn

        except Exception as e:
            self.logger.error(f"Failed to create rule {name}: {e}")
            raise

    def attach_lambda_target(
        self,
        rule_name: str,
        lambda_function_name: str,
        input_data: Dict,
        dlq_arn: str = None
    ):
        """
        Attach Lambda function to rule.

        Args:
            rule_name: EventBridge rule name
            lambda_function_name: Lambda function name
            input_data: JSON input to Lambda
            dlq_arn: Dead-letter queue ARN (optional)
        """
        lambda_arn = f'arn:aws:lambda:{self.region}:{self.account_id}:function:{lambda_function_name}'

        # Build target configuration
        target_config = {
            'Id': '1',
            'Arn': lambda_arn,
            'Input': json.dumps(input_data),
            'RetryPolicy': {
                'MaximumRetryAttempts': 2,
                'MaximumEventAge': 3600
            }
        }

        # Add DLQ if provided
        if dlq_arn:
            target_config['DeadLetterConfig'] = {'Arn': dlq_arn}

        # Attach target
        try:
            response = self.events_client.put_targets(
                Rule=rule_name,
                Targets=[target_config]
            )

            if response['FailedEntryCount'] > 0:
                raise Exception(f"Failed to attach target: {response['FailedEntries']}")

            self.logger.info(f"Attached Lambda {lambda_function_name} to rule {rule_name}")

        except Exception as e:
            self.logger.error(f"Error attaching target: {e}")
            raise

        # Grant permission
        self._grant_lambda_permission(rule_name, lambda_function_name)

    def _grant_lambda_permission(self, rule_name: str, function_name: str):
        """Grant EventBridge permission to invoke Lambda."""
        rule_arn = f'arn:aws:events:{self.region}:{self.account_id}:rule/{rule_name}'

        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'AllowEventBridge{rule_name}',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=rule_arn
            )
            self.logger.info(f"Granted permission for {rule_name} to invoke {function_name}")

        except self.lambda_client.exceptions.ResourceConflictException:
            # Permission already exists
            self.logger.info(f"Permission already exists for {function_name}")

    def create_dlq(self) -> str:
        """
        Create Dead-Letter Queue for failed events.

        Returns:
            DLQ ARN
        """
        try:
            # Create queue
            queue_response = self.sqs_client.create_queue(
                QueueName='EventBridgeDLQ',
                Attributes={
                    'MessageRetentionPeriod': '1209600'  # 14 days
                }
            )

            # Get ARN
            queue_url = queue_response['QueueUrl']
            attrs = self.sqs_client.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['QueueArn']
            )
            dlq_arn = attrs['Attributes']['QueueArn']

            self.logger.info(f"Created DLQ: {dlq_arn}")
            return dlq_arn

        except self.sqs_client.exceptions.QueueNameExists:
            # Queue already exists, get ARN
            queue_url = self.sqs_client.get_queue_url(QueueName='EventBridgeDLQ')['QueueUrl']
            attrs = self.sqs_client.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['QueueArn']
            )
            dlq_arn = attrs['Attributes']['QueueArn']
            self.logger.info(f"Using existing DLQ: {dlq_arn}")
            return dlq_arn

    def setup_daily_tick_collection(self, dlq_arn: str = None):
        """
        Set up daily tick collection (weekdays at 6 AM ET).

        Args:
            dlq_arn: Dead-letter queue ARN (optional)
        """
        rule_name = 'DailyTickCollectorRule'

        # Create rule
        self.create_rule(
            name=rule_name,
            schedule='cron(0 11 ? * MON-FRI *)',  # 6 AM ET, weekdays
            description='Trigger daily stock data collection at 6 AM ET (11 AM UTC). '
                       'Collects OHLCV data for all 5,000 symbols from previous trading day.'
        )

        # Attach Lambda
        self.attach_lambda_target(
            rule_name=rule_name,
            lambda_function_name='DailyTickCollector',
            input_data={
                'trigger': 'scheduled',
                'data_type': 'daily_ticks',
                'collection_date': 'auto',
                'symbols': 'all',
                's3_bucket': 'us-equity-datalake'
            },
            dlq_arn=dlq_arn
        )

        self.logger.info("Daily tick collection configured")

    def setup_monthly_fundamental_refresh(self, dlq_arn: str = None):
        """
        Set up monthly fundamental refresh (first Sunday at 2 AM ET).

        Args:
            dlq_arn: Dead-letter queue ARN (optional)
        """
        rule_name = 'MonthlyFundamentalRefreshRule'

        # Create rule
        self.create_rule(
            name=rule_name,
            schedule='cron(0 7 ? * SUN#1 *)',  # 2 AM ET, first Sunday
            description='Refresh fundamental data first Sunday of each month at 2 AM ET. '
                       'Fetches quarterly filings from SEC EDGAR for all symbols.'
        )

        # Attach Lambda
        self.attach_lambda_target(
            rule_name=rule_name,
            lambda_function_name='FundamentalCollector',
            input_data={
                'trigger': 'scheduled',
                'data_type': 'fundamentals',
                'refresh_type': 'full',
                'source': 'sec_edgar'
            },
            dlq_arn=dlq_arn
        )

        self.logger.info("Monthly fundamental refresh configured")

    def deploy_all(self):
        """Deploy all EventBridge rules for data lake automation."""
        self.logger.info("Deploying EventBridge rules for US Equity Data Lake...")

        # Create DLQ
        dlq_arn = self.create_dlq()

        # Setup all rules
        self.setup_daily_tick_collection(dlq_arn=dlq_arn)
        self.setup_monthly_fundamental_refresh(dlq_arn=dlq_arn)

        self.logger.info("✅ All EventBridge rules deployed successfully!")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    manager = DataLakeEventBridge(region='us-east-2')
    manager.deploy_all()
```

**Output:**
```
Deploying EventBridge rules for US Equity Data Lake...
Created DLQ: arn:aws:sqs:us-east-2:123456789012:EventBridgeDLQ
Created rule: DailyTickCollectorRule
Attached Lambda DailyTickCollector to rule DailyTickCollectorRule
Granted permission for DailyTickCollectorRule to invoke DailyTickCollector
Daily tick collection configured
Created rule: MonthlyFundamentalRefreshRule
Attached Lambda FundamentalCollector to rule MonthlyFundamentalRefreshRule
Granted permission for MonthlyFundamentalRefreshRule to invoke FundamentalCollector
Monthly fundamental refresh configured
✅ All EventBridge rules deployed successfully!
```

---

## Summary

### Core EventBridge Methods Used in Data Lake

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `put_rule()` | Create scheduled rule | Define when Lambda should run (cron expression) |
| `put_targets()` | Attach Lambda to rule | Tell EventBridge which Lambda to invoke |
| `add_permission()` | Grant invoke permission | Allow EventBridge to call Lambda |

### Key Parameters

- **ScheduleExpression**: `cron(0 11 ? * MON-FRI *)` = weekdays at 6 AM ET
- **RetryPolicy**: `MaximumRetryAttempts=2`, `MaximumEventAge=3600` (fail fast)
- **DeadLetterConfig**: SQS queue ARN to capture failed events
- **Input**: JSON configuration passed to Lambda event

### Common Cron Expressions

```python
'cron(0 11 ? * MON-FRI *)'   # Weekdays at 6 AM ET
'cron(0 7 ? * SUN#1 *)'      # First Sunday of month at 2 AM ET
'cron(0 11 * * ? *)'         # Every day at 6 AM ET
'cron(0 */6 * * ? *)'        # Every 6 hours
```

### Next Steps

1. Review [LAMBDA_TUTORIAL.md](LAMBDA_TUTORIAL.md) to write Lambda functions
2. Deploy EventBridge rules using the `DataLakeEventBridge` class
3. Monitor rule executions in CloudWatch Logs
4. Set up CloudWatch alarms on DLQ message count
