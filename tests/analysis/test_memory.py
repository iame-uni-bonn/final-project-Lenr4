import numpy as np
import pandas as pd
import pytest

from lennart_epp.analysis.memory import (
    _compute_autocovariance,
    _compute_mean,
    _compute_variance,
    check_stat_diff_close,
    compute_acf,
    compute_hurst_exponent,
)


@pytest.fixture
def test_series():
    """Generate a random time series for testing."""
    rng = np.random.default_rng(2837)
    return rng.standard_normal(100)


@pytest.fixture
def test_df(test_series):
    """Generate a DataFrame containing the test series."""
    return pd.DataFrame({"close_price": test_series})


@pytest.fixture
def precomputed_values(test_series):
    """Precompute mean and variance tests."""
    mean_series = np.mean(test_series)
    variance_series = np.sum((test_series - mean_series) ** 2)
    return mean_series, variance_series


def test_compute_mean(test_series):
    """Test whether _compute_mean returns the correct mean."""
    assert _compute_mean(test_series) == pytest.approx(np.mean(test_series), rel=1e-6)


def test_compute_variance(test_series, precomputed_values):
    """Test whether _compute_variance returns the correct variance."""
    mean_series, expected_variance = precomputed_values
    assert _compute_variance(test_series, mean_series) == pytest.approx(
        expected_variance, rel=1e-6
    )


def test_compute_autocovariance_lag_1(test_series, precomputed_values):
    """Test whether _compute_autocovariance returns the correct value for lag=1."""
    mean_series, _ = precomputed_values
    lag = 1
    expected_autocov = np.sum(
        (test_series[lag:] - mean_series)
        * (test_series[: len(test_series) - lag] - mean_series)
    )
    assert _compute_autocovariance(test_series, mean_series, lag) == pytest.approx(
        expected_autocov, rel=1e-6
    )


def test_compute_autocovariance_lag_5(test_series, precomputed_values):
    """Test whether _compute_autocovariance returns the correct value for lag=10."""
    mean_series, _ = precomputed_values
    lag = 10
    expected_autocov = np.sum(
        (test_series[lag:] - mean_series)
        * (test_series[: len(test_series) - lag] - mean_series)
    )
    assert _compute_autocovariance(test_series, mean_series, lag) == pytest.approx(
        expected_autocov, rel=1e-6
    )


def test_compute_acf_output_structure(test_df):
    """Test whether compute_acf returns a dictionary with 'acf' and 'lags' keys."""
    result = compute_acf(test_df, column="close_price", lags=10)
    assert all(key in result for key in ("acf", "lags"))


expected_length = 11


def test_compute_acf_length(test_df):
    """Test whether compute_acf returns arrays of correct length."""
    result = compute_acf(test_df, column="close_price", lags=10)
    assert len(result["acf"]) == expected_length
    assert len(result["lags"]) == expected_length


def test_compute_hurst_exponent_output(test_df):
    """Test if compute_hurst_exponent returns dictionary with 'Hurst Exponent' key."""
    result = compute_hurst_exponent(test_df, column="close_price")
    assert isinstance(result, dict)
    assert "Hurst Exponent" in result


def test_compute_hurst_exponent_range(test_df):
    """Test whether the computed Hurst Exponent is within the expected range [0,1]."""
    hurst_value = compute_hurst_exponent(test_df, column="close_price")[
        "Hurst Exponent"
    ]
    assert 0 <= hurst_value <= 1


def test_check_stat_diff_close_returns_dict(test_df):
    """Ensure check_stat_diff_close returns a dictionary."""
    result = check_stat_diff_close(test_df, column="close_price")
    assert isinstance(result, dict)


def test_check_stat_diff_close_has_expected_keys(test_df):
    """Ensure the returned dictionary has the correct keys."""
    result = check_stat_diff_close(test_df, column="close_price")
    expected_keys = {"ADF Test Statistic", "P-Value", "Is Stationary"}
    assert set(result.keys()) == expected_keys


def test_check_stat_diff_close_raises_error_for_missing_column(test_df):
    """Ensure a ValueError is raised if the column does not exist."""
    with pytest.raises(ValueError, match="Column .* not found in dataframe."):
        check_stat_diff_close(test_df, column="non_existent_column")
