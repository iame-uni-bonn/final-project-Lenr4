import numpy as np
import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model


def _compute_residuals(df: pd.DataFrame, model_results: dict) -> np.ndarray:
    """Compute residuals for an autoregressive (AR) model.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'close_price' column.
        model_results (dict): A dictionary containing the fitted AR parameters.

    Returns:
        np.ndarray: An array of residuals.
    """
    fitted_values = _predict_ar(df, model_results)
    return df["close_price"].iloc[len(df) - len(fitted_values) :] - fitted_values


def _calculate_aic(residuals: np.ndarray, p: int) -> float:
    """Calculate the Akaike Information Criterion (AIC) for model evaluation.

    Args:
        residuals (np.ndarray): An array of residuals from the AR model.
        p (int): The number of autoregressive parameters in the model.

    Returns:
        float: The computed AIC value.
    """
    n = len(residuals)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + 2 * (p + 1)


def _calculate_bic(residuals: np.ndarray, p: int) -> float:
    """Calculate the Bayesian Information Criterion (BIC) for model evaluation.

    Args:
        residuals (np.ndarray): An array of residuals from the AR model.
        p (int): The number of autoregressive parameters in the model.

    Returns:
        float: The computed BIC value.
    """
    n = len(residuals)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + np.log(n) * (p + 1)


def _predict_ar(df: pd.DataFrame, model_results: dict) -> np.ndarray:
    """Generate 1 step predictions (fitted values) using an autoregressive (AR) model.

    The function applies AR model coefficients to past values of 'close_price'
    to make predictions.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'close_price' column.
        model_results (dict): A dictionary containing model coefficients under
                              'integrated_coefficients'.

    Returns:
        np.ndarray: An array of predicted values based on the AR model.
    """
    integrated_coeff = model_results["integrated_coefficients"][
        "coefficient"
    ].to_numpy()
    lags = len(integrated_coeff) - 1
    predictions = []
    for i in range(lags, len(df)):
        prediction = integrated_coeff[0] + sum(
            integrated_coeff[j] * df["close_price"].iloc[i - j]
            for j in range(1, lags + 1)
        )
        predictions.append(prediction)
    return np.array(predictions)


def evaluate_ar_models(
    df: pd.DataFrame, max_p: int = 15, criterion: str = "aic"
) -> dict:
    """Evaluate multiple Autoregressive (AR) models and select the best ones.

    This function fits AR models for different lag values (p) up to `max_p`,
    computes the AIC and BIC scores for each model, and returns the top-performing
    models based on the selected criterion.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'close_price' column.
        max_p (int, optional): The maximum number of autoregressive lags to test.
                               Defaults to 15.
        criterion (str, optional): The selection criterion for ranking models.
                                   Can be "aic" or "bic". Defaults to "aic".

    Returns:
        dict: A dictionary containing:
            - **top_models** (list[dict]): The top 3 sorted by the given criterion.
            - **model_metrics** (list[dict]): A list of evaluated models with metrics.
            - **metadata** (dict): Information about evaluation, `max_p`, `criterion`.
    """
    results = []

    for p in range(1, max_p + 1):
        model_results = fit_ar_model(df, column="close_price", p=p)
        residuals = _compute_residuals(df, model_results)
        aic = _calculate_aic(residuals, p)
        bic = _calculate_bic(residuals, p)

        results.append(
            {
                "p": p,
                "aic": aic,
                "bic": bic,
                "p_value": model_results["p_value"],
                "differenced": model_results["differenced"],
                "coefficients": model_results["coefficients"],
                "integrated_coefficients": model_results["integrated_coefficients"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=criterion, ascending=True)

    top_models = results_df.head(3).to_dict(orient="records")
    model_metrics = [dict(model) for model in results]

    return {
        "top_models": top_models,
        "model_metrics": model_metrics,
        "metadata": {"max_p": max_p, "criterion": criterion},
    }
