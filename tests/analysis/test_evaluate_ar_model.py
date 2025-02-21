import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess

from lennart_epp.analysis.evaluate_ar_model import (
    _calculate_aic,
    _calculate_bic,
    _compute_residuals,
    _predict_ar,
)

Max_Aic_Bic = 1e6


@pytest.fixture
def test_dataframe():
    rng = np.random.default_rng(934)

    ar_params = [1, -0.75, 0.25]
    ma_params = [1]

    ar_process = ArmaProcess(ar_params, ma_params)
    simulated_data = ar_process.generate_sample(
        nsample=200, distrvs=rng.standard_normal
    )

    df = pd.DataFrame({"close_price": simulated_data})
    return df


@pytest.fixture
def test_model_results(test_dataframe):
    model = AutoReg(test_dataframe["close_price"], lags=2, old_names=False).fit()
    integrated_coefficients = pd.DataFrame(
        {"coefficient": np.insert(model.params.values, 0, model.params.iloc[0])}
    )

    return {
        "p_value": 0.05,
        "differenced": False,
        "coefficients": model.params.to_numpy(),
        "integrated_coefficients": integrated_coefficients,
    }


def test_compute_residuals(test_dataframe, test_model_results):
    residuals = _compute_residuals(test_dataframe, test_model_results)

    assert len(residuals) == len(test_dataframe) - (
        len(test_model_results["integrated_coefficients"]) - 1
    )
    assert isinstance(residuals, pd.Series)
    assert not np.isnan(residuals).all()


def test_calculate_aic_output(test_dataframe, test_model_results):
    residuals = _compute_residuals(test_dataframe, test_model_results)
    aic = _calculate_aic(residuals, p=2)

    assert isinstance(aic, float)
    assert aic < Max_Aic_Bic


def test_calculate_aic_correctness(test_dataframe, test_model_results):
    residuals = _compute_residuals(test_dataframe, test_model_results)

    n = len(residuals)

    p = 2

    sigma_squared = np.var(residuals, ddof=1)

    expected_aic = 2 * p + n * np.log(sigma_squared)

    computed_aic = _calculate_aic(residuals, p)
    assert np.isclose(computed_aic, expected_aic, atol=1.5), (
        f"AIC does not match: expected {expected_aic}, got {computed_aic}"
    )


def test_calculate_bic_output(test_dataframe, test_model_results):
    residuals = _compute_residuals(test_dataframe, test_model_results)
    bic = _calculate_bic(residuals, p=2)

    assert isinstance(bic, float)
    assert bic < Max_Aic_Bic


def test_calculate_bic_correctness(test_dataframe, test_model_results):
    residuals = _compute_residuals(test_dataframe, test_model_results)

    n = len(residuals)

    p = 2

    sigma_squared = np.var(residuals, ddof=1)

    expected_bic = p * np.log(n) + n * np.log(sigma_squared)

    computed_bic = _calculate_bic(residuals, p)

    assert np.isclose(computed_bic, expected_bic, atol=5.0), (
        f"BIC does not match: expected {expected_bic}, got {computed_bic}"
    )


def test_predict_ar(test_dataframe, test_model_results):
    predictions = _predict_ar(test_dataframe, test_model_results)

    assert len(predictions) == len(test_dataframe) - (
        len(test_model_results["integrated_coefficients"]) - 1
    )
    assert isinstance(predictions, np.ndarray)
    assert not np.isnan(predictions).all()
