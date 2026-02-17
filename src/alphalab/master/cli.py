"""CLI for SecurityMaster operations"""
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os

from alphalab.master.security_master import SecurityMaster
from alphalab.utils.logger import setup_logger

load_dotenv()

# Source parquet bundled with the package
SOURCE_PATH = Path(__file__).resolve().parent.parent / "data" / "security_master.parquet"


def _get_working_path() -> Path:
    """Get working copy path under LOCAL_STORAGE_PATH."""
    base = os.getenv("LOCAL_STORAGE_PATH", "")
    if not base:
        return SOURCE_PATH
    return Path(os.path.expanduser(base)) / "data" / "meta" / "master" / "security_master.parquet"


def _copy_source_to_working(source: Path, working: Path) -> None:
    """Copy source parquet to working path if needed."""
    if source == working:
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Source parquet not found at {source}. "
            "Run: python scripts/build_security_master.py"
        )
    working.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(working))


def main():
    parser = argparse.ArgumentParser(description="SecurityMaster operations")
    parser.add_argument(
        '--build',
        action='store_true',
        help='Copy source parquet to working path, update from SEC/Nasdaq/OpenFIGI/yfinance'
    )

    args = parser.parse_args()

    logger = setup_logger(
        name="master.cli",
        log_dir=Path("data/logs/master"),
        console_output=True
    )

    working_path = _get_working_path()

    if args.build:
        logger.info("Building working copy from source parquet...")
        _copy_source_to_working(SOURCE_PATH, working_path)
        master = SecurityMaster(local_path=working_path)
        result = master.update()
        master.save_local(working_path)
        logger.info(f"Build complete: {result}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
