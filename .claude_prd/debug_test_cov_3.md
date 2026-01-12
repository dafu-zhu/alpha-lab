# Failures

=================================== FAILURES ===================================
________ TestS3Client.test_create_boto_config_with_all_optional_params _________
tests/unit/storage/test_s3_client.py:58: in test_create_boto_config_with_all_optional_params
    assert boto_config.s3 is not None
E   assert None is not None
E    +  where None = <botocore.config.Config object at 0x7fe2129e61b0>.s3
___________ TestS3Client.test_us_east_1_regional_endpoint_parameter ____________
tests/unit/storage/test_s3_client.py:98: in test_us_east_1_regional_endpoint_parameter
    assert boto_config.s3['us_east_1_regional_endpoint'] == 'legacy'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'NoneType' object is not subscriptable
_____ TestUploadApp.test_upload_derived_fundamental_overwrite_log_message ______
tests/unit/storage/test_upload_app.py:1951: in test_upload_derived_fundamental_overwrite_log_message
    app.upload_derived_fundamental("2024-01-01", "2024-12-31", max_workers=1, overwrite=False)
src/quantdl/storage/app.py:1204: in upload_derived_fundamental
    executor.submit(
/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/unittest/mock.py:1139: in __call__
    return self._mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/unittest/mock.py:1143: in _mock_call
    return self._execute_mock_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/unittest/mock.py:1204: in _execute_mock_call
    result = effect(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/storage/test_upload_app.py:1941: in submit_side_effect
    future.result.return_value = fn(*args, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^
src/quantdl/storage/app.py:1071: in _process_symbol_derived_fundamental
    derived_df, derived_reason = self.data_collectors.collect_derived_long(
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: cannot unpack non-iterable Mock object
================================ tests coverage ================================

# Task

Scan relevant source code. Distinguish source code error and test code error. Make a plan to debug in multi-phase. Implement the debugging plan.

Note: Keep all interaction and commit message extremely concise. Sacrifice grammar for the sake of concision. 

When debug finished, create a github issue with your plan, including the checked off items listed. Use Github CLI to operate with github.