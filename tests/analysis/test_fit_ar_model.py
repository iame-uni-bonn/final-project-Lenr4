import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from lennart_epp.analysis.fit_ar_model import (
    _ar_model,
    _check_stationarity,
    _create_lagged_features,
    _difference_series,
)

significane = 0.05
expected_coeff_count = 3  # For p=2: Intercept + 2 Lags
expected_length_lagged = 3


def test_check_stationarity():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"price": rng.normal(0, 1, 100)})
    stationary, p_value = _check_stationarity(df, "price", significance=significane)
    assert bool(stationary) is True

    # 2. Non-stationary series
    df["price"] = np.linspace(1, 100, 100) + rng.normal(0, 0.5, 100)
    stationary, p_value = _check_stationarity(df, "price", significance=significane)
    assert bool(stationary) is False
    assert p_value > significane


def test_difference_series():
    df = pd.DataFrame({"price": [100, 101, 103, 106]})
    result = _difference_series(df.copy(), "price")
    assert "diff_price" in result.columns
    # Check that the second value of the differenced series equals 1.
    assert result["diff_price"].iloc[1] == 1


def test_create_lagged_features():
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50]})
    result = _create_lagged_features(df.copy(), "price", 2)
    assert "price_lag1" in result.columns
    assert "price_lag2" in result.columns
    assert len(result) == expected_length_lagged


def test_ar_model_output():
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

    assert len(coeffs) == expected_coeff_count
    assert isinstance(coeffs, np.ndarray)


def test_ar_model_correctness():
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
