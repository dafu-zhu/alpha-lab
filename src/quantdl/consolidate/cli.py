"""CLI for year consolidation."""


def main():
    """Main entry point for year consolidation."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate year's monthly files into history.parquet")
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to consolidate (e.g., 2025)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite if year already exists in history.parquet'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/storage.yaml',
        help='Path to configuration file (default: configs/storage.yaml)'
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from quantdl.update.app import DailyUpdateApp

    # Initialize app
    app = DailyUpdateApp(config_path=args.config)

    # Run consolidation
    print(f"Consolidating year {args.year} (force={args.force})")
    stats = app.consolidate_year(year=args.year, force=args.force)

    print(f"\nConsolidation completed:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == '__main__':
    main()
