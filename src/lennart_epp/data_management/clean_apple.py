import pandas as pd


def _select_and_rename_column(df: pd.DataFrame) -> pd.DataFrame:
    """Select the Close column and rename it to close_price."""
    return df[["Close"]].rename(columns={"Close": "close_price"})


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by forward and backward filling."""
    return df.ffill().bfill()


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on the index."""
    return df[~df.index.duplicated(keep="first")]


def _convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the entire DataFrame to numeric types, rounding to 2 decimal places."""
    return df.apply(pd.to_numeric, errors="coerce").round(2).astype("float32")


def _validate_dataframe(df: pd.DataFrame):
    """Ensure the DataFrame contains the necessary columns."""
    missing_col_msg = "The DataFrame does not contain a 'Close' column."
    if "Close" not in df.columns:
        raise ValueError(missing_col_msg)


def _convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Date column to a datetime index."""
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def clean_apple_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw data and returns the cleaned DataFrame."""
    _validate_dataframe(df)

    df = _convert_to_datetime(df)
    df = _select_and_rename_column(df)
    df = _handle_missing_values(df)
    df = _remove_duplicates(df)
    df = _convert_to_numeric(df)

    return df
