import numpy as np
import pandas as pd
import pytest

from lennart_epp.analysis.forecast_ar import forecast_ar_multi_step


@pytest.fixture
def test_dataframe():
    rng = np.random.default_rng(83)
    dates = pd.date_range(start="2023-01-01", periods=400, freq="D")
    data = np.linspace(100, 200, num=400) + rng.normal(scale=5, size=400)
    return pd.DataFrame({"close_price": data}, index=dates)


@pytest.fixture
def test_integrated_coefficients():
    coeffs = np.array([0.5, 0.3, -0.2])
    return pd.DataFrame({"coefficient": coeffs})


def test_forecast_ar_multi_step_output_type(
    test_dataframe, test_integrated_coefficients
):
    """Test that forecast_ar_multi_step returns a pandas Series."""
    forecast_steps = 10
    forecasts = forecast_ar_multi_step(
        test_dataframe, test_integrated_coefficients, forecast_steps
    )

    assert isinstance(forecasts, pd.Series)


def test_forecast_ar_multi_step_length(test_dataframe, test_integrated_coefficients):
    """Test that the forecast length matches the requested number of steps."""
    forecast_steps = 10
    forecasts = forecast_ar_multi_step(
        test_dataframe, test_integrated_coefficients, forecast_steps
    )

    assert len(forecasts) == forecast_steps


def test_forecast_ar_multi_step_no_nans(test_dataframe, test_integrated_coefficients):
    """Test that the forecast does not contain NaN values."""
    forecast_steps = 10
    forecasts = forecast_ar_multi_step(
        test_dataframe, test_integrated_coefficients, forecast_steps
    )

    assert not forecasts.isna().any()


def test_forecast_ar_multi_step_index(test_dataframe, test_integrated_coefficients):
    """Test that the forecast index matches the expected date range."""
    forecast_steps = 10
    forecasts = forecast_ar_multi_step(
        test_dataframe, test_integrated_coefficients, forecast_steps
    )

    expected_index_start = test_dataframe.index[-344]
    expected_index = pd.date_range(
        start=expected_index_start, periods=forecast_steps, freq="D"
    )

    assert forecasts.index.equals(expected_index), "Index does not match"
