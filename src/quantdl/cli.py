from quantdl.storage.upload_app import UploadApp

def main() -> None:
    app = UploadApp()
    try:
        app.run(
            start_year=2010,
            end_year=2025,
            max_workers=20,
            sleep_time=0.05,
            overwrite=True,
            run_fundamental=True,
            run_derived_fundamental=True,
            run_ttm_fundamental=True,
        )
    finally:
        app.close()