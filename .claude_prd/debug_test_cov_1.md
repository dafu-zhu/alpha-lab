## Error Infos from Pytest

```
______________________________________________________ TestRecentDailyTicksChunking.test_recent_daily_ticks_chunk_size_zero ______________________________________________________
tests\unit\collection\test_crsp_ticks.py:1387: in test_recent_daily_ticks_chunk_size_zero
    assert mock_conn.raw_sql.call_count == 1
E   AssertionError: assert 0 == 1
E    +  where 0 = <Mock name='mock.raw_sql' id='2456140098896'>.call_count
E    +    where <Mock name='mock.raw_sql' id='2456140098896'> = <Mock id='2456138537520'>.raw_sql
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Resolving symbols: 100%|██████████| 1/1 [00:00<?, ?sym/s]
____________________________________________________ TestRecentDailyTicksChunking.test_recent_daily_ticks_chunk_size_negative ____________________________________________________
tests\unit\collection\test_crsp_ticks.py:1465: in test_recent_daily_ticks_chunk_size_negative
    assert mock_conn.raw_sql.call_count == 1
E   AssertionError: assert 0 == 1
E    +  where 0 = <Mock name='mock.raw_sql' id='2456141762432'>.call_count
E    +    where <Mock name='mock.raw_sql' id='2456141762432'> = <Mock id='2456141074352'>.raw_sql
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Resolving symbols: 100%|██████████| 1/1 [00:00<?, ?sym/s]
__________________________________________________ TestRecentDailyTicksChunking.test_recent_daily_ticks_empty_frames_unadjusted __________________________________________________
tests\unit\collection\test_crsp_ticks.py:1521: in test_recent_daily_ticks_empty_frames_unadjusted
    query = mock_conn.raw_sql.call_args[0][0]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Resolving symbols: 100%|██████████| 1/1 [00:00<?, ?sym/s]
____________________________________________________ TestRecentDailyTicksChunking.test_recent_daily_ticks_many_failed_symbols ____________________________________________________
tests\unit\collection\test_crsp_ticks.py:1570: in test_recent_daily_ticks_many_failed_symbols
    assert len(warning_calls) >= 1
E   assert 0 >= 1
E    +  where 0 = len([])
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Resolving symbols: 100%|██████████| 25/25 [00:00<00:00, 6037.40sym/s]
__________________________________________________________ TestCIKResolver.test_get_cik_continues_when_security_id_none __________________________________________________________
tests\unit\storage\test_cik_resolver.py:175: in test_get_cik_continues_when_security_id_none
    assert cik == "0000320193"
E   AssertionError: assert '320193' == '0000320193'
E
E     - 0000320193
E     ? ----
E     + 320193
__________________________________________________________ TestCIKResolver.test_get_cik_continues_when_cik_record_empty __________________________________________________________
tests\unit\storage\test_cik_resolver.py:200: in test_get_cik_continues_when_cik_record_empty
    assert cik == "0000320193"
E   AssertionError: assert '320193' == '0000320193'
E
E     - 0000320193
E     ? ----
E     + 320193
_________________________________________________________ TestS3Client.test_create_boto_config_with_all_optional_params __________________________________________________________
tests\unit\storage\test_s3_client.py:44: in test_create_boto_config_with_all_optional_params
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
____________________________________________________________ TestS3Client.test_us_east_1_regional_endpoint_parameter _____________________________________________________________
tests\unit\storage\test_s3_client.py:86: in test_us_east_1_regional_endpoint_parameter
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
_________________________________________________________ TestS3Client.test_request_min_compression_size_bytes_parameter _________________________________________________________
tests\unit\storage\test_s3_client.py:117: in test_request_min_compression_size_bytes_parameter
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
____________________________________________________________ TestS3Client.test_disable_request_compression_parameter _____________________________________________________________
tests\unit\storage\test_s3_client.py:148: in test_disable_request_compression_parameter
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
____________________________________________________________ TestS3Client.test_request_checksum_calculation_parameter ____________________________________________________________
tests\unit\storage\test_s3_client.py:179: in test_request_checksum_calculation_parameter
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
____________________________________________________________ TestS3Client.test_response_checksum_validation_parameter ____________________________________________________________
tests\unit\storage\test_s3_client.py:210: in test_response_checksum_validation_parameter
    client = S3Client(config_path=config_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
_____________________________________________________________ TestS3Client.test_client_property_returns_boto_client ______________________________________________________________
tests\unit\storage\test_s3_client.py:239: in test_client_property_returns_boto_client
    s3_client = S3Client(config_path=config_path)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:17: in __init__
    self.boto_config = self._create_boto_config()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
src\quantdl\storage\s3_client.py:39: in _create_boto_config
    if 'connect_timeout' in client_cfg:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: argument of type 'Mock' is not iterable
______________________________________________________ TestUploadApp.test_upload_daily_ticks_monthly_alpaca_canceled_status ______________________________________________________
tests\unit\storage\test_upload_app.py:1635: in test_upload_daily_ticks_monthly_alpaca_canceled_status
    app.upload_daily_ticks(2025, use_monthly_partitions=True, by_year=False, chunk_size=2, sleep_time=0.0)
src\quantdl\storage\app.py:229: in upload_daily_ticks
    result = self.data_publishers.publish_daily_ticks(
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1200: in _execute_mock_call
    result = next(effect)
             ^^^^^^^^^^^^
E   StopIteration
______________________________________________________ TestUploadApp.test_upload_daily_ticks_monthly_alpaca_skipped_status _______________________________________________________
tests\unit\storage\test_upload_app.py:1661: in test_upload_daily_ticks_monthly_alpaca_skipped_status
    app.upload_daily_ticks(2025, use_monthly_partitions=True, by_year=False, chunk_size=2, sleep_time=0.0)
src\quantdl\storage\app.py:229: in upload_daily_ticks
    result = self.data_publishers.publish_daily_ticks(
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1200: in _execute_mock_call
    result = next(effect)
             ^^^^^^^^^^^^
E   StopIteration
_________________________________________________________ TestUploadApp.test_upload_fundamental_company_unknown_fallback _________________________________________________________
tests\unit\storage\test_upload_app.py:1878: in test_upload_fundamental_company_unknown_fallback
    assert any("Unknown" in c for c in info_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_fundamental_company_unknown_fallback.<locals>.<genexpr> at 0x0000023C1FCA2DC0>)
____________________________________________________ TestUploadApp.test_upload_ttm_fundamental_large_symbols_without_cik_list ____________________________________________________
tests\unit\storage\test_upload_app.py:1906: in test_upload_ttm_fundamental_large_symbols_without_cik_list
    assert any("showing first 30" in c for c in info_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_ttm_fundamental_large_symbols_without_cik_list.<locals>.<genexpr> at 0x0000023C1FCA2330>)
______________________________________________________ TestUploadApp.test_upload_derived_fundamental_overwrite_log_message _______________________________________________________
tests\unit\storage\test_upload_app.py:1943: in test_upload_derived_fundamental_overwrite_log_message
    assert any("already exists" in c and "continuing to refresh" in c for c in info_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_derived_fundamental_overwrite_log_message.<locals>.<genexpr> at 0x0000023C200AF440>)
_______________________________________________________ TestUploadApp.test_upload_derived_fundamental_large_non_sec_filers _______________________________________________________
tests\unit\storage\test_upload_app.py:1973: in test_upload_derived_fundamental_large_non_sec_filers
    assert any("showing first 30" in c for c in info_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_derived_fundamental_large_non_sec_filers.<locals>.<genexpr> at 0x0000023BDDC54D40>)
_______________________________________________________________ TestUploadApp.test_upload_all_data_sets_all_flags ________________________________________________________________
tests\unit\storage\test_upload_app.py:1987: in test_upload_all_data_sets_all_flags
    app.upload_all_data(2024, 2024, run_all=True)
    ^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'UploadApp' object has no attribute 'upload_all_data'
____________________________________________________________ TestValidator.test_list_available_years_fundamental_type ____________________________________________________________
tests\unit\storage\test_validator.py:29: in test_list_available_years_fundamental_type
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
________________________________________________________________ TestValidator.test_list_available_years_ttm_type ________________________________________________________________
tests\unit\storage\test_validator.py:51: in test_list_available_years_ttm_type
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
________________________________________________________ TestValidator.test_list_available_years_with_continuation_token _________________________________________________________
tests\unit\storage\test_validator.py:84: in test_list_available_years_with_continuation_token
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
_________________________________________________________ TestValidator.test_list_available_years_fundamental_continues __________________________________________________________
tests\unit\storage\test_validator.py:108: in test_list_available_years_fundamental_continues
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
________________________________________________________________ TestValidator.test_data_exists_monthly_partition ________________________________________________________________
tests\unit\storage\test_validator.py:122: in test_data_exists_monthly_partition
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
________________________________________________________________ TestValidator.test_top_3000_exists_error_logging ________________________________________________________________
tests\unit\storage\test_validator.py:143: in test_top_3000_exists_error_logging
    validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket", logger=mock_logger)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Validator.__init__() got an unexpected keyword argument 'logger'
```

## Task 

Run Plan mode on debugging in phases, and then implement the plan