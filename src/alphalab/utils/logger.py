import logging
import sys
import datetime as dt
from pathlib import Path
from typing import Optional


def console_log(logger: logging.Logger, message: str, section: bool = False):
    """Log a message with optional section header formatting."""
    if section:
        logger.info(f"{'─' * 60}")
        logger.info(f"  {message}")
        logger.info(f"{'─' * 60}")
    else:
        logger.info(message)


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


def setup_logger(
    name: str,
    log_dir: str | Path = "data/logs",
    level: int = logging.WARNING,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    daily_rotation: bool = True,
    console_output: bool = False
) -> logging.Logger:
    """
    Setup and configure a logger with file handler and optional console output.

    :param name: Logger name (e.g., 'fundamental.AAPL')
    :param log_dir: Directory to store log files (default: 'data/logs')
    :param level: Logging level (default: logging.WARNING)
    :param log_format: Log message format string
    :param daily_rotation: If True, creates separate log file per day (default: True)
    :param console_output: If True, also outputs logs to console (default: False)

    :return: Configures a logger where:
    1. FILE receives logs at `file_level` (default WARNING) and above.
    2. CONSOLE (if enabled) receives ONLY INFO logs (blocks Warnings/Errors).

    Example:
        >>> logger = setup_logger('fundamental.AAPL', 'data/logs/fundamental')
        >>> logger.warning('Failed to fetch data')

        >>> # With console output enabled
        >>> logger = setup_logger('ticks.daily', console_output=True)
        >>> logger.info('Processing started')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup file handler with optional daily rotation
    if daily_rotation:
        log_date = dt.datetime.now().strftime('%Y-%m-%d')
        log_file = log_path / f"logs_{log_date}.log"
    else:
        log_file = log_path / f"{name.replace('.', '_')}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if console_output:
        # Use sys.stdout for INFO (standard output), reserve stderr for actual errors if needed
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set to INFO so it captures progress
        console_handler.setLevel(logging.INFO)
        
        # CRITICAL FIX: Add the filter to BLOCK Warnings/Errors from Console
        # This ensures the console stays "clean"
        console_handler.addFilter(MaxLevelFilter(logging.INFO))
        
        # Optional: Simpler format for console (just the message)
        console_formatter = logging.Formatter('%(message)s') 
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)

    return logger


class LoggerFactory:
    """
    Factory class for creating loggers with consistent configuration.
    Useful when you need to create multiple loggers with the same settings.

    Example:
        >>> factory = LoggerFactory(log_dir='data/logs/fundamental', level=logging.INFO)
        >>> logger1 = factory.get_logger('fundamental.AAPL')
        >>> logger2 = factory.get_logger('fundamental.TSLA')
    """

    def __init__(
        self,
        log_dir: str | Path = "data/logs",
        level: int = logging.WARNING,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        daily_rotation: bool = True,
        console_output: bool = False
    ):
        """
        Initialize logger factory with default configuration.

        Args:
            log_dir: Directory to store log files
            level: Logging level
            log_format: Log message format string
            daily_rotation: Enable daily log file rotation
            console_output: Enable console output
        """
        self.log_dir = log_dir
        self.level = level
        self.log_format = log_format
        self.daily_rotation = daily_rotation
        self.console_output = console_output

    def get_logger(self, name: str) -> logging.Logger:
        """
        Create a logger with the factory's configuration.

        :param name: Logger name

        :return: Configured logger instance
        """
        return setup_logger(
            name=name,
            log_dir=self.log_dir,
            level=self.level,
            log_format=self.log_format,
            daily_rotation=self.daily_rotation,
            console_output=self.console_output
        )
