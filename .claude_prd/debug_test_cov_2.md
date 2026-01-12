# Failures

=================================== FAILURES ===================================
_____ TestRecentDailyTicksChunking.test_recent_daily_ticks_chunk_size_zero _____
tests/unit/collection/test_crsp_ticks.py:1388: in test_recent_daily_ticks_chunk_size_zero
    assert mock_raw_sql_with_retry.call_count == 1
E   AssertionError: assert 0 == 1
E    +  where 0 = <MagicMock name='raw_sql_with_retry' id='139976477894624'>.call_count
----------------------------- Captured stderr call -----------------------------

Resolving symbols:   0%|          | 0/1 [00:00<?, ?sym/s]
Resolving symbols: 100%|██████████| 1/1 [00:00<00:00, 25575.02sym/s]
___ TestRecentDailyTicksChunking.test_recent_daily_ticks_chunk_size_negative ___
tests/unit/collection/test_crsp_ticks.py:1467: in test_recent_daily_ticks_chunk_size_negative
    assert mock_raw_sql_with_retry.call_count == 1
E   AssertionError: assert 0 == 1
E    +  where 0 = <MagicMock name='raw_sql_with_retry' id='139976481392160'>.call_count
----------------------------- Captured stderr call -----------------------------

Resolving symbols:   0%|          | 0/1 [00:00<?, ?sym/s]
Resolving symbols: 100%|██████████| 1/1 [00:00<00:00, 23831.27sym/s]
_ TestRecentDailyTicksChunking.test_recent_daily_ticks_empty_frames_unadjusted _
tests/unit/collection/test_crsp_ticks.py:1524: in test_recent_daily_ticks_empty_frames_unadjusted
    query = mock_raw_sql_with_retry.call_args[0][1]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
----------------------------- Captured stderr call -----------------------------

Resolving symbols:   0%|          | 0/1 [00:00<?, ?sym/s]
Resolving symbols: 100%|██████████| 1/1 [00:00<00:00, 23696.63sym/s]
___ TestRecentDailyTicksChunking.test_recent_daily_ticks_many_failed_symbols ___
tests/unit/collection/test_crsp_ticks.py:1574: in test_recent_daily_ticks_many_failed_symbols
    assert len(warning_calls) >= 1
E   assert 0 >= 1
E    +  where 0 = len([])
----------------------------- Captured stderr call -----------------------------

Resolving symbols:   0%|          | 0/25 [00:00<?, ?sym/s]
Resolving symbols: 100%|██████████| 25/25 [00:00<00:00, 8111.52sym/s]
________ TestS3Client.test_create_boto_config_with_all_optional_params _________
tests/unit/storage/test_s3_client.py:48: in test_create_boto_config_with_all_optional_params
    assert mock_boto_client.called
E   AssertionError: assert False
E    +  where False = <MagicMock name='client' id='139976479862528'>.called
___________ TestS3Client.test_us_east_1_regional_endpoint_parameter ____________
tests/unit/storage/test_s3_client.py:90: in test_us_east_1_regional_endpoint_parameter
    call_kwargs = mock_boto_client.call_args[1]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
________ TestS3Client.test_request_min_compression_size_bytes_parameter ________
tests/unit/storage/test_s3_client.py:122: in test_request_min_compression_size_bytes_parameter
    call_kwargs = mock_boto_client.call_args[1]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
___________ TestS3Client.test_disable_request_compression_parameter ____________
tests/unit/storage/test_s3_client.py:154: in test_disable_request_compression_parameter
    call_kwargs = mock_boto_client.call_args[1]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
___________ TestS3Client.test_request_checksum_calculation_parameter ___________
tests/unit/storage/test_s3_client.py:186: in test_request_checksum_calculation_parameter
    call_kwargs = mock_boto_client.call_args[1]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
___________ TestS3Client.test_response_checksum_validation_parameter ___________
tests/unit/storage/test_s3_client.py:218: in test_response_checksum_validation_parameter
    call_kwargs = mock_boto_client.call_args[1]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
________ TestUploadApp.test_upload_fundamental_company_unknown_fallback ________
tests/unit/storage/test_upload_app.py:1879: in test_upload_fundamental_company_unknown_fallback
    assert any("Unknown" in c for c in all_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_fundamental_company_unknown_fallback.<locals>.<genexpr> at 0x7f4e391c5560>)
___ TestUploadApp.test_upload_ttm_fundamental_large_symbols_without_cik_list ___
tests/unit/storage/test_upload_app.py:1908: in test_upload_ttm_fundamental_large_symbols_without_cik_list
    assert any("showing first 30" in c for c in all_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_ttm_fundamental_large_symbols_without_cik_list.<locals>.<genexpr> at 0x7f4e391c6330>)
_____ TestUploadApp.test_upload_derived_fundamental_overwrite_log_message ______
tests/unit/storage/test_upload_app.py:1946: in test_upload_derived_fundamental_overwrite_log_message
    assert any("already exists" in c and "continuing to refresh" in c for c in all_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_derived_fundamental_overwrite_log_message.<locals>.<genexpr> at 0x7f4e391c7440>)
______ TestUploadApp.test_upload_derived_fundamental_large_non_sec_filers ______
tests/unit/storage/test_upload_app.py:1977: in test_upload_derived_fundamental_large_non_sec_filers
    assert any("showing first 30" in c for c in all_calls)
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_derived_fundamental_large_non_sec_filers.<locals>.<genexpr> at 0x7f4e38e54520>)
_______ TestValidator.test_list_available_years_with_continuation_token ________
tests/unit/storage/test_validator.py:91: in test_list_available_years_with_continuation_token
    assert 2024 in years and 2023 in years
E   assert (2024 in [1])
_______________ TestValidator.test_top_3000_exists_error_logging _______________
/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/unittest/mock.py:918: in assert_called
    raise AssertionError(msg)
E   AssertionError: Expected 'error' to have been called.

During handling of the above exception, another exception occurred:
tests/unit/storage/test_validator.py:147: in test_top_3000_exists_error_logging
    mock_logger.error.assert_called()
E   AssertionError: Expected 'error' to have been called.
------------------------------ Captured log call -------------------------------
ERROR    validation:validation.py:191 Error checking data/symbols/2024/06/top3000.txt: An error occurred (403) when calling the HeadObject operation: Forbidden
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.12-final-0 _______________

Coverage XML written to file coverage.xml

# Task

Scan the relevant source code, understand the logic. Distinguish debugging on source code and on tests. Implement the plan. 

Note: In all interaction and commit messages, be extremely concise and sacrifice grammar for the sake of concision