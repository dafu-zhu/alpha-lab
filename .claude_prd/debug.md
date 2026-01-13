==================================================================================== FAILURES ====================================================================================
___________________________________________________________ TestDataPublishers.test_publish_daily_ticks_success_yearly ___________________________________________________________
tests\unit\storage\test_data_publishers.py:60: in test_publish_daily_ticks_success_yearly
    assert result == {"symbol": "AAPL", "status": "success", "error": None}
E   AssertionError: assert {'error': 'ca...mbol': 'AAPL'} == {'error': Non...mbol': 'AAPL'}
E
E     Omitting 1 identical items, use -vv to show
E     Differing items:
E     {'status': 'failed'} != {'status': 'success'}
E     {'error': 'catching classes that do not inherit from BaseException is not allowed'} != {'error
E
E     ...Full output truncated (12 lines hidden), use '-vv' to show
__________________________________________________________ TestDataPublishers.test_publish_daily_ticks_metadata_yearly ___________________________________________________________
tests\unit\storage\test_data_publishers.py:157: in test_publish_daily_ticks_metadata_yearly
    assert result["status"] == "success"
E   AssertionError: assert 'failed' == 'success'
E
E     - success
E     + failed
_________________________________________________________ TestDataPublishers.test_publish_daily_ticks_value_error_skips __________________________________________________________
tests\unit\storage\test_data_publishers.py:281: in test_publish_daily_ticks_value_error_skips
    assert result["status"] == "skipped"
E   AssertionError: assert 'failed' == 'skipped'
E
E     - skipped
E     + failed
_________________________________________________________ TestDataPublishers.test_publish_daily_ticks_value_error_other __________________________________________________________
tests\unit\storage\test_data_publishers.py:821: in test_publish_daily_ticks_value_error_other
    assert result["error"] == "some other error"
E   AssertionError: assert 'catching cla...s not allowed' == 'some other error'
E
E     - some other error
E     + catching classes that do not inherit from BaseException is not allowed
_____________________________________________________________ TestUploadApp.test_upload_daily_ticks_success_monthly ______________________________________________________________
tests\unit\storage\test_upload_app.py:128: in test_upload_daily_ticks_success_monthly
    assert app.data_collectors.collect_daily_ticks_month.call_count == 24
E   AssertionError: assert 0 == 24
E    +  where 0 = <Mock name='mock.collect_daily_ticks_month' id='2401133644880'>.call_count
E    +    where <Mock name='mock.collect_daily_ticks_month' id='2401133644880'> = <Mock id='2401132322576'>.collect_daily_ticks_month
E    +      where <Mock id='2401132322576'> = <quantdl.storage.app.UploadApp object at 0x0000022F0EC8AC00>.data_collectors
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Uploading 2024 symbols: 100%|██████████| 2/2 [00:00<?, ?sym/s]
_________________________________________________________________ TestUploadApp.test_upload_daily_ticks_by_year __________________________________________________________________
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:960: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\storage\test_upload_app.py:171: in test_upload_daily_ticks_by_year
    app.data_collectors.collect_daily_ticks_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2024)
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to be called once. Called 0 times.
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Uploading 2024 symbols: 100%|██████████| 2/2 [00:00<00:00, 1995.86sym/s]
___________________________________________________ TestUploadApp.test_upload_daily_ticks_monthly_alpaca_calls_bulk_with_sleep ___________________________________________________
tests\unit\storage\test_upload_app.py:271: in test_upload_daily_ticks_monthly_alpaca_calls_bulk_with_sleep
    assert any(
E   assert False
E    +  where False = any(<generator object TestUploadApp.test_upload_daily_ticks_monthly_alpaca_calls_bulk_with_sleep.<locals>.<genexpr> at 0x0000022F0EBA6DC0>)
___________________________________________________ TestUploadApp.test_upload_daily_ticks_monthly_alpaca_failed_status_counts ____________________________________________________
tests\unit\storage\test_upload_app.py:303: in test_upload_daily_ticks_monthly_alpaca_failed_status_counts
    assert len(completed_logs) > 0
E   assert 0 > 0
E    +  where 0 = len([])
___________________________________________________________ TestUploadApp.test_upload_daily_ticks_monthly_bulk_alpaca ____________________________________________________________
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:1020: in assert_any_call
    raise AssertionError(
E   AssertionError: collect_daily_ticks_month_bulk(['AAPL'], 2025, 1, sleep_time=0.5) call not found

During handling of the above exception, another exception occurred:
tests\unit\storage\test_upload_app.py:887: in test_upload_daily_ticks_monthly_bulk_alpaca
    app.data_collectors.collect_daily_ticks_month_bulk.assert_any_call(
E   AssertionError: collect_daily_ticks_month_bulk(['AAPL'], 2025, 1, sleep_time=0.5) call not found
__________________________________________________________ TestUploadApp.test_upload_daily_ticks_alpaca_by_year_ignored __________________________________________________________
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:910: in assert_not_called
    raise AssertionError(msg)
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to not have been called. Called 1 times.
E   Calls: [call(['AAPL'], 2025), call().get('AAPL', shape: (0, 0)
E   ┌┐
E   ╞╡
E   └┘)].

During handling of the above exception, another exception occurred:
tests\unit\storage\test_upload_app.py:912: in test_upload_daily_ticks_alpaca_by_year_ignored
    app.data_collectors.collect_daily_ticks_year_bulk.assert_not_called()
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to not have been called. Called 1 times.
E   Calls: [call(['AAPL'], 2025), call().get('AAPL', shape: (0, 0)
E   ┌┐
E   ╞╡
E   └┘)].
E
E   pytest introspection follows:
E
E   Args:
E   assert (['AAPL'], 2025) == ()
E
E     Left contains 2 more items, first extra item: ['AAPL']
E
E     Full diff:
E     - ()
E     + (
E     +     [...
E
E     ...Full output truncated (4 lines hidden), use '-vv' to show
____________________________________________________________ TestUploadApp.test_upload_daily_ticks_by_year_crsp_bulk _____________________________________________________________
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:960: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\storage\test_upload_app.py:1569: in test_upload_daily_ticks_by_year_crsp_bulk
    app.data_collectors.collect_daily_ticks_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2024)
E   AssertionError: Expected 'collect_daily_ticks_year_bulk' to be called once. Called 0 times.
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Uploading 2024 symbols: 100%|██████████| 2/2 [00:00<00:00, 1993.49sym/s]
_____________________________________________________ TestUploadApp.test_upload_daily_ticks_by_year_checks_any_month_exists ______________________________________________________
..\..\..\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\Lib\unittest\mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'publish_daily_ticks' to have been called once. Called 2 times.
E   Calls: [call('AAPL', 2024, <Mock name='mock.security_master.get_security_id()' id='2401132030496'>, <Mock name='mock.collect_daily_ticks_year()' id='2401132028720'>, by_year=False),
E    call('MSFT', 2024, <Mock name='mock.security_master.get_security_id()' id='2401132030496'>, <Mock name='mock.collect_daily_ticks_year()' id='2401132028720'>, by_year=False)].

During handling of the above exception, another exception occurred:
tests\unit\storage\test_upload_app.py:1595: in test_upload_daily_ticks_by_year_checks_any_month_exists
    app.data_publishers.publish_daily_ticks.assert_called_once()
E   AssertionError: Expected 'publish_daily_ticks' to have been called once. Called 2 times.
E   Calls: [call('AAPL', 2024, <Mock name='mock.security_master.get_security_id()' id='2401132030496'>, <Mock name='mock.collect_daily_ticks_year()' id='2401132028720'>, by_year=False),
E    call('MSFT', 2024, <Mock name='mock.security_master.get_security_id()' id='2401132030496'>, <Mock name='mock.collect_daily_ticks_year()' id='2401132028720'>, by_year=False)].
E
E   pytest introspection follows:
E
E   Args:
E   assert ('MSFT', 2024...01132028720'>) == ()
E
E     Left contains 4 more items, first extra item: 'MSFT'
E
E     Full diff:
E     - ()
E     + (
E     +     'MSFT',...
E
E     ...Full output truncated (4 lines hidden), use '-vv' to show
E   Kwargs:
E   assert {'by_year': False} == {}
E
E     Left contains 1 more item:
E     {'by_year': False}
E
E     Full diff:
E     - {}
E     + {
E     +     'by_year': False,
E     + }
------------------------------------------------------------------------------ Captured stderr call ------------------------------------------------------------------------------
Uploading 2024 symbols: 100%|██████████| 2/2 [00:00<?, ?sym/s]
___________________________________________________ TestDailyUpdateAppNoWRDSUpdateMinuteTicks.test_update_minute_ticks_success ___________________________________________________
tests\unit\update\test_app_no_wrds.py:746: in test_update_minute_ticks_success
    assert stats['success'] == 1
E   assert 0 == 1