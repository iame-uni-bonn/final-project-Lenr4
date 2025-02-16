import numpy as np
import pandas as pd
from arch.unitroot import ADF
from numpy.linalg import inv


def _check_stationarity(
    df: pd.DataFrame, column: str, significance: float = 0.05
) -> tuple[bool, float]:
    adf_test = ADF(df[column].dropna())
    p_value = adf_test.pvalue

    return p_value < significance, p_value


def _difference_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[f"diff_{column}"] = df[column].diff().dropna()
    return df


def _create_lagged_features(df: pd.DataFrame, column: str, p: int) -> pd.DataFrame:
    for lag in range(1, p + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)

    df_lagged = df.dropna()

    return df_lagged


def _ar_model(df: pd.DataFrame, column: str, p: int) -> np.ndarray:
    x = df[[f"{column}_lag{i}" for i in range(1, p + 1)]].to_numpy()
    y = df[column].to_numpy()

    x_with_intercept = np.column_stack((np.ones(x.shape[0]), x))

    coefficients = inv(x_with_intercept.T @ x_with_intercept) @ x_with_intercept.T @ y

    return coefficients


def _integrate_ar_coefficients(
    diff_coefficients: np.ndarray, *, differenced: bool
) -> pd.DataFrame:
    if not differenced:
        integrated_coeff = diff_coefficients
    else:
        integrated_coeff = np.zeros(len(diff_coefficients) + 1)  # Platz für AR(p+1)
        integrated_coeff[0] = diff_coefficients[0]  # Intercept bleibt gleich

        # Erster AR-Koeffizient (vom differenzierten Model)
        integrated_coeff[1] = 1 + diff_coefficients[1]

        # Nachfolgende AR-Koeffizienten
        for i in range(2, len(diff_coefficients)):
            integrated_coeff[i] = diff_coefficients[i - 1] - diff_coefficients[i]

        # Zusätzliches Lag (durch Integration)
        integrated_coeff[-1] = -diff_coefficients[-1]

    # DataFrame für besseren Überblick
    integrated_coeff_df = pd.DataFrame(
        {
            "coefficient": integrated_coeff,
            "lag": [
                f"Lag {i}" if i > 0 else "Intercept"
                for i in range(len(integrated_coeff))
            ],
        }
    )

    return integrated_coeff_df


def fit_ar_model(df: pd.DataFrame, column: str = "close_price", p: int = 3) -> dict:
    """Fitte ein AR(p)-Modell und speichere differenzierte & originale Koeffizienten."""
    is_stationary, p_value = _check_stationarity(df, column)
    differenced = False

    if not is_stationary:
        df = _difference_series(df, column)
        column = f"diff_{column}"
        differenced = True

    df = _create_lagged_features(df, column, p)

    diff_coefficients = _ar_model(df, column, p)

    integrated_coefficients = _integrate_ar_coefficients(
        diff_coefficients, differenced=differenced
    )

    return {
        "coefficients": diff_coefficients,
        "integrated_coefficients": integrated_coefficients,
        "lag_order": p,
        "p_value": p_value,
        "differenced": differenced,
    }
