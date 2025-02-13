from pathlib import Path

import yfinance as yf

from lennart_epp.config import SRC


def download_apple_data():
    raw_data_path = SRC / "data" / "apple_data.csv"

    # Ensure the directory exists
    Path(raw_data_path.parent).mkdir(parents=True, exist_ok=True)

    apple_data = yf.download("AAPL", start="2013-01-01", end="2023-12-31")

    apple_data.to_csv(raw_data_path)
