import numpy as np
import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model


def _compute_residuals(df: pd.DataFrame, model_results: dict) -> np.ndarray:
    fitted_values = _predict_ar(df, model_results)
    return df["close_price"].iloc[len(df) - len(fitted_values) :] - fitted_values


def _calculate_aic(residuals: np.ndarray, p: int) -> float:
    n = len(residuals)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + 2 * (p + 1)


def _calculate_bic(residuals: np.ndarray, p: int) -> float:
    n = len(residuals)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + np.log(n) * (p + 1)


def _predict_ar(df: pd.DataFrame, model_results: dict) -> np.ndarray:
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
