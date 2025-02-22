import pandas as pd

from lennart_epp.config import BLD, SRC
from lennart_epp.data_management.clean_apple import clean_apple_data
from lennart_epp.data_management.download_apple import download_apple_data


def task_download_apple_data(
    script=SRC / "data_management" / "download_apple.py",
    produces=SRC / "data" / "apple_data.csv",
):
    """Download and save historical Apple stock data.

    Args:
        script (Path, optional): Path to the script handling the download.
        produces (Path, optional): Path where the downloaded CSV file is stored.

    Returns:
        None

    Raises:
        AssertionError: If the file was not successfully created.
    """
    download_apple_data()
    assert produces.exists(), f"Failed to produce {produces}"


def task_clean_apple_data(
    script=SRC / "data_management" / "clean_apple.py",
    data=SRC / "data" / "apple_data.csv",
    produces=BLD / "data" / "cleaned_apple_data.pkl",
):
    """Load, clean, and store Apple stock data.

    Args:
        script (Path, optional): Path to the script handling data cleaning.
        data (Path, optional): Path to the raw stock data CSV file.
        produces (Path, optional): Path where the cleaned data is stored.

    Returns:
        None

    Raises:
        AssertionError: If the cleaned data file was not successfully created.
    """
    df = pd.read_csv(data, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df = clean_apple_data(df)
    df.to_pickle(produces)
    assert produces.exists(), f"Failed to produce {produces}"
