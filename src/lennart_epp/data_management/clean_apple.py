import pandas as pd


def _select_and_rename_column(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 'Close' column and rename it to 'close_price'.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with a single column named 'close_price'.
    """
    return df[["Close"]].rename(columns={"Close": "close_price"})


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using forward and backward filling.

    Args:
        df (pd.DataFrame): Input DataFrame that may contain missing values.

    Returns:
        pd.DataFrame: DataFrame with no missing values.
    """
    return df.ffill().bfill()


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on the index.

    Args:
        df (pd.DataFrame): Input DataFrame that may contain duplicate index entries.

    Returns:
        pd.DataFrame: DataFrame without duplicate index entries.
    """
    return df[~df.index.duplicated(keep="first")]


def _convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns in the DataFrame to numeric types.

    Args:
        df (pd.DataFrame): Input DataFrame containing numeric data.

    Returns:
        pd.DataFrame: DataFrame with all values converted to numeric types.
    """
    return df.apply(pd.to_numeric, errors="coerce").round(2).astype("float32")


def _validate_dataframe(df: pd.DataFrame):
    """Validate that the DataFrame contains the required columns.

    Args:
        df (pd.DataFrame): Input DataFrame to validate.

    Raises:
        ValueError: If the 'Close' column is not found in the DataFrame.
    """
    missing_col_msg = "The DataFrame does not contain a 'Close' column."
    if "Close" not in df.columns:
        raise ValueError(missing_col_msg)


def _convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'Date' column to a datetime index.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Date' column.

    Returns:
        pd.DataFrame: DataFrame with a datetime index.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def clean_apple_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw Apple stock data.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis.

    Raises:
        ValueError: If the required 'Close' column is missing in the input DataFrame.
    """
    _validate_dataframe(df)

    df = _convert_to_datetime(df)
    df = _select_and_rename_column(df)
    df = _handle_missing_values(df)
    df = _remove_duplicates(df)
    df = _convert_to_numeric(df)

    return df
