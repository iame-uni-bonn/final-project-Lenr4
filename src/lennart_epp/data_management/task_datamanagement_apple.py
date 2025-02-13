"""Tasks for managing the stock data."""

from lennart_epp.config import SRC
from lennart_epp.data_management.download_apple import download_apple_data


def task_download_apple_data(
    script=SRC / "data_management" / "download_apple.py",
    produces=SRC / "data" / "apple_data.csv",
):
    """Download AAPL stock data."""
    download_apple_data()
    assert produces.exists(), f"Failed to produce {produces}"
