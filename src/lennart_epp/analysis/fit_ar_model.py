import numpy as np
import pandas as pd
from arch.unitroot import ADF
from numpy.linalg import inv


def _check_stationarity(
    df: pd.DataFrame, column: str, significance: float = 0.05
) -> tuple[bool, float]:
    # Run the ADF test
    adf_test = ADF(df[column].dropna())
    p_value = adf_test.pvalue

    # Return stationarity status and p-value
    return p_value < significance, p_value


def _difference_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[f"diff_{column}"] = df[column].diff().dropna()
    return df


def _create_lagged_features(df: pd.DataFrame, column: str, p: int) -> pd.DataFrame:
    # Generate lagged features
    for lag in range(1, p + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)

    # Drop missing values and assign to a new DataFrame
    df_lagged = df.dropna()

    return df_lagged


def _ar_model(df: pd.DataFrame, column: str, p: int) -> np.ndarray:
    # Prepare the feature matrix (X) and target vector (y)
    x = df[[f"{column}_lag{i}" for i in range(1, p + 1)]].to_numpy()
    y = df[column].to_numpy()

    x_with_intercept = np.column_stack((np.ones(x.shape[0]), x))

    # Calculate coefficients using the OLS formula: (X'X)^(-1)X'y
    coefficients = inv(x_with_intercept.T @ x_with_intercept) @ x_with_intercept.T @ y

    return coefficients


def fit_ar_model(df: pd.DataFrame, column: str = "close_price", p: int = 3) -> dict:
    is_stationary, p_value = _check_stationarity(df, column)
    differenced = False

    if not is_stationary:
        df = _difference_series(df, column)
        column = f"diff_{column}"
        differenced = True

    df = _create_lagged_features(df, column, p)

    coefficients = _ar_model(df, column, p)

    return {
        "coefficients": coefficients,
        "lag_order": p,
        "p_value": p_value,
        "differenced": differenced,
    }
