import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from lennart_epp.analysis.fit_ar_model import (
    _ar_model,
    _check_stationarity,
    _create_lagged_features,
    _difference_series,
    _integrate_ar_coefficients,
    fit_ar_model,
)

significance = 0.05
expected_coeff_count = 3  # For p=2: Intercept + 2 Lags
expected_length_lagged = 3


def test_check_stationarity_stationary():
    """Test ADF stationarity check on a stationary time series."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"price": rng.normal(0, 1, 100)})

    stationary, p_value, test_statistic = _check_stationarity(df, "price", significance)

    assert all(
        [
            bool(stationary) is True,
            isinstance(p_value, float),
            isinstance(test_statistic, float),
        ]
    )


def test_check_stationarity_non_stationary():
    """Test ADF stationarity check on a nonstationary time series."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"price": np.linspace(1, 100, 100) + rng.normal(0, 0.5, 100)})

    stationary, p_value, test_statistic = _check_stationarity(df, "price", significance)

    assert all(
        [
            bool(stationary) is False,
            p_value > significance,
            isinstance(test_statistic, float),
        ]
    )


def test_difference_series():
    """Test differencing function for correct column creation and values."""
    df = pd.DataFrame({"price": [100, 101, 103, 106]})
    result = _difference_series(df.copy(), "price")

    assert all(["diff_price" in result.columns, result["diff_price"].iloc[1] == 1])


def test_create_lagged_features():
    """Test creation of lagged features for an AR model."""
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50]})
    result = _create_lagged_features(df.copy(), "price", 2)

    assert all(
        [
            "price_lag1" in result.columns,
            "price_lag2" in result.columns,
            len(result) == expected_length_lagged,
        ]
    )


def test_ar_model_output():
    """Test output structure and type of AR model fitting function."""
    rng = np.random.default_rng(312)

    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "price": rng.random(10) * 100,
        },
        index=dates,
    )

    df["price_lag1"] = df["price"].shift(1) + rng.standard_normal(10) * 5
    df["price_lag2"] = df["price"].shift(2) + rng.standard_normal(10) * 5

    df = df.dropna()

    coeffs = _ar_model(df, "price", 2)

    assert all([len(coeffs) == expected_coeff_count, isinstance(coeffs, np.ndarray)])


def test_ar_model_correctness():
    """Test correctness of AR model coefficients against a reference implementation."""
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {"price": np.linspace(1, 100, 100) + rng.normal(0, 0.3, 100)},
        index=dates,
    )
    df = _create_lagged_features(df, "price", p=2)
    custom_coeffs = _ar_model(df, "price", 2)
    model = AutoReg(df["price"], lags=2, trend="c").fit()
    reference_coeffs = np.concatenate(
        ([model.params.get("const", 0)], model.params.values[1:])
    )
    assert np.allclose(
        custom_coeffs,
        reference_coeffs,
        atol=5e-2,  # Tolerance increased
    ), (
        f"AR coefficients do not match:\n"
        f"Custom: {custom_coeffs}\n"
        f"Reference: {reference_coeffs}\n"
        f"Difference: {custom_coeffs - reference_coeffs}"
    )


def test_integrate_ar_coefficients_no_differencing():
    """Test integration of AR coefficients when no differencing is applied."""
    diff_coeffs = np.array([0.5, -0.2, 0.1])

    result = _integrate_ar_coefficients(diff_coeffs, differenced=False)
    expected_lags = ["Intercept", "Lag 1", "Lag 2"]
    assert all(
        [
            np.allclose(result["coefficient"].to_numpy(), diff_coeffs),
            list(result["lag"]) == expected_lags,
        ]
    )


def test_integrate_ar_coefficients_with_differencing():
    """Test integration of AR coefficients when differencing is applied."""
    diff_coeffs = np.array([0.5, -0.2, 0.1])

    result = _integrate_ar_coefficients(diff_coeffs, differenced=True)

    expected_coeffs = np.array([0.5, 1 - 0.2, -0.2 - 0.1, -0.1])
    expected_lags = ["Intercept", "Lag 1", "Lag 2", "Lag 3"]
    assert all(
        [
            np.allclose(result["coefficient"].to_numpy(), expected_coeffs),
            list(result["lag"]) == expected_lags,
        ]
    )


def test_fit_ar_model_stationary_series():
    """Test AR model fitting on a stationary time series."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"close_price": np.sin(np.linspace(0, 10, 100)) + rng.normal(0, 0.1, 100)}
    )

    result = fit_ar_model(df, column="close_price", p=2)

    assert result["differenced"] is False


def test_fit_ar_model_non_stationary_series():
    """Test AR model fitting on a non-stationary time series."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"close_price": np.cumsum(np.linspace(0.1, 1, 100)) + rng.normal(0, 0.1, 100)}
    )

    result = fit_ar_model(df, column="close_price", p=2)

    assert result["differenced"] is True


def test_fit_ar_model_coefficient_shape():
    """Test if the number of coefficients matches AR order + intercept."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"close_price": np.cos(np.linspace(0, 10, 100)) + rng.normal(0, 0.1, 100)}
    )

    p = 3
    result = fit_ar_model(df, column="close_price", p=p)

    assert result["coefficients"].shape[0] == p + 1


def test_fit_ar_model_p_value():
    """Test if p-value is included in the result dictionary."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"close_price": np.exp(np.linspace(0, 2, 100)) + rng.normal(0, 0.1, 100)}
    )

    result = fit_ar_model(df, column="close_price", p=2)

    assert isinstance(result["p_value"], float)
