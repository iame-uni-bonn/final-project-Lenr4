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
    evaluate_ar_models,
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
    """Test that _compute_residuals returns a valid non-NaN pandas Series."""
    residuals = _compute_residuals(test_dataframe, test_model_results)

    assert all(
        [
            isinstance(residuals, pd.Series),
            len(residuals)
            == len(test_dataframe)
            - (len(test_model_results["integrated_coefficients"]) - 1),
            not np.isnan(residuals).all(),
        ]
    )


def test_calculate_aic_output(test_dataframe, test_model_results):
    """Test that _calculate_aic returns a valid float within a reasonable range."""
    residuals = _compute_residuals(test_dataframe, test_model_results)
    aic = _calculate_aic(residuals, p=2)

    assert all([isinstance(aic, float), aic < Max_Aic_Bic])


def test_calculate_aic_correctness(test_dataframe, test_model_results):
    """Test that _calculate_aic computes a nearly equal value to the formula."""
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
    """Test that _calculate_bic returns a valid float within a reasonable range."""
    residuals = _compute_residuals(test_dataframe, test_model_results)
    bic = _calculate_bic(residuals, p=2)

    assert all([isinstance(bic, float), bic < Max_Aic_Bic])


def test_calculate_bic_correctness(test_dataframe, test_model_results):
    """Test that _calculate_bic computes a nearly equal value to the formula."""
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
    """Test that _predict_ar produces valid non-NaN numpy predictions."""
    predictions = _predict_ar(test_dataframe, test_model_results)

    assert all(
        [
            len(predictions)
            == len(test_dataframe)
            - (len(test_model_results["integrated_coefficients"]) - 1),
            isinstance(predictions, np.ndarray),
            not np.isnan(predictions).all(),
        ]
    )


def test_evaluate_ar_models_output_structure(test_dataframe):
    """Test that evaluate_ar_models returns a dictionary with expected keys."""
    results = evaluate_ar_models(test_dataframe, max_p=5, criterion="aic")
    assert set(results.keys()) == {"top_models", "model_metrics", "metadata"}


def test_evaluate_ar_models_top_models_length(test_dataframe):
    """Test that 'top_models' contains up to 3 models."""
    count_top_models = 3
    results = evaluate_ar_models(test_dataframe, max_p=5, criterion="aic")
    assert len(results["top_models"]) <= count_top_models


def test_evaluate_ar_models_model_metrics_length(test_dataframe):
    """Test that 'model_metrics' contains max_p models."""
    max_p = 5
    results = evaluate_ar_models(test_dataframe, max_p=max_p, criterion="aic")
    assert len(results["model_metrics"]) == max_p


def test_evaluate_ar_models_metadata(test_dataframe):
    """Test that 'metadata' contains correct max_p and criterion values."""
    max_p = 5
    criterion = "aic"
    results = evaluate_ar_models(test_dataframe, max_p=max_p, criterion=criterion)
    assert results["metadata"] == {"max_p": max_p, "criterion": criterion}


def test_evaluate_ar_models_top_models_sorted(test_dataframe):
    """Test that 'top_models' are sorted by the chosen criterion."""
    results = evaluate_ar_models(test_dataframe, max_p=5, criterion="aic")
    sorted_aic_values = [model["aic"] for model in results["top_models"]]
    assert sorted_aic_values == sorted(sorted_aic_values)


def test_evaluate_ar_models_model_keys(test_dataframe):
    """Test that each model result contains expected keys."""
    results = evaluate_ar_models(test_dataframe, max_p=5, criterion="aic")
    required_keys = {"p", "aic", "bic", "p_value", "differenced", "coefficients"}

    for model in results["model_metrics"]:
        assert set(model.keys()).issuperset(required_keys)


def test_evaluate_ar_models_coefficients_type(test_dataframe):
    """Test that 'coefficients' in model results are numpy arrays."""
    results = evaluate_ar_models(test_dataframe, max_p=5, criterion="aic")

    for model in results["model_metrics"]:
        assert isinstance(model["coefficients"], np.ndarray)
