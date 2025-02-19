import numpy as np
import pandas as pd
import pytest

from lennart_epp.data_management.clean_apple import (
    _convert_to_datetime,
    _convert_to_numeric,
    _handle_missing_values,
    _remove_duplicates,
    _select_and_rename_column,
    _validate_dataframe,
)


@pytest.fixture
def raw_data():
    data = {
        "Date": [
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-03",
            "2022-01-04",
        ],
        "Close": [100, np.nan, 102, 102, 105],
        "Other": ["A", "B", "C", "C", "D"],
    }
    return pd.DataFrame(data)


def test_select_and_rename_column(raw_data):
    result = _select_and_rename_column(raw_data)
    assert list(result.columns) == ["close_price"]
    pd.testing.assert_series_equal(
        result["close_price"], raw_data["Close"], check_names=False
    )


expected_value_missing = 100


def test_handle_missing_values(raw_data):
    result = _handle_missing_values(raw_data)
    assert result.isna().sum().sum() == 0
    assert result.loc[1, "Close"] == expected_value_missing


expected_length_index = 4


def test_remove_duplicates(raw_data):
    df = raw_data.copy().set_index("Date")
    result = _remove_duplicates(df)
    assert result.index.duplicated().sum() == 0
    assert len(result) == expected_length_index


def test_convert_to_numeric(raw_data):
    df = raw_data.copy().astype(str)
    result = _convert_to_numeric(df)
    for col in result.columns:
        assert result[col].dtype == "float32"
    np.testing.assert_almost_equal(result.loc[0, "Close"], 100.0, decimal=2)


def test_validate_dataframe(raw_data):
    _validate_dataframe(raw_data)
    df_missing = raw_data.drop(columns=["Close"])
    with pytest.raises(
        ValueError, match="The DataFrame does not contain a 'Close' column."
    ):
        _validate_dataframe(df_missing)


def test_convert_to_datetime(raw_data):
    result = _convert_to_datetime(raw_data.copy())
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0] == pd.Timestamp("2022-01-01")
