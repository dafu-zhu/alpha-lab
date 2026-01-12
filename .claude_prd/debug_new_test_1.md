## Failure

______________________________________________________ TestGetSymbolsWithRecentFilings.test_get_symbols_with_recent_filings ______________________________________________________
tests\unit\update\test_app.py:431: in test_get_symbols_with_recent_filings
    result = app.get_symbols_with_recent_filings(
src\quantdl\update\app.py:233: in get_symbols_with_recent_filings
    symbol_to_cik = {sym: cik for sym, cik in symbol_to_cik.items() if cik is not None}
                                              ^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'Mock' object is not iterable
_____________________________________________ TestGetSymbolsWithRecentFilings.test_get_symbols_with_recent_filings_progress_logging ______________________________________________
tests\unit\update\test_app.py:480: in test_get_symbols_with_recent_filings_progress_logging
    result = app.get_symbols_with_recent_filings(
src\quantdl\update\app.py:233: in get_symbols_with_recent_filings
    symbol_to_cik = {sym: cik for sym, cik in symbol_to_cik.items() if cik is not None}
                                              ^^^^^^^^^^^^^^^^^^^^^
E   TypeError: 'Mock' object is not iterable

## Task

Debug