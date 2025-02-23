import numpy as np
import pandas as pd
from arch.unitroot import ADF
from numpy.linalg import inv


def _check_stationarity(
    df: pd.DataFrame, column: str, significance: float = 0.05
) -> tuple[bool, float, float]:
    """Perform the Augmented Dickey-Fuller (ADF) test to check stationarity.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        column (str): The name of the column to test for stationarity.
        significance (float, optional): The significance level for the test.

    Returns:
        tuple:
            - bool: True if the series is stationary
            - float: The p-value from the ADF test.
            - float: The ADF test statistic.
    """
    adf_test = ADF(df[column].dropna())
    p_value = adf_test.pvalue
    test_statistic_adf = adf_test.stat

    return p_value < significance, p_value, test_statistic_adf


def _difference_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply first differencing to a column in a DataFrame.

    Args:
        df (pd.DataFrame): The dataframe containing the time series.
        column (str): The column to be differenced.

    Returns:
        pd.DataFrame: A new DataFrame with the differenced column.
    """
    df_copy = df.copy()
    df_copy[f"diff_{column}"] = df_copy[column].diff().dropna()
    return df_copy[[f"diff_{column}"]]


def _create_lagged_features(df: pd.DataFrame, column: str, p: int) -> pd.DataFrame:
    """Generate lagged features for an autoregressive model.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        column (str): The column for which lagged features should be created.
        p (int): The number of lagged periods to generate.

    Returns:
        pd.DataFrame: A DataFrame with the original column and its lagged features.
    """
    for lag in range(1, p + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)

    df_lagged = df.dropna()

    return df_lagged


def _ar_model(df: pd.DataFrame, column: str, p: int) -> np.ndarray:
    """Estimate autoregressive (AR) model parameters using the least squares method.

    Args:
        df (pd.DataFrame): The DataFrame with time series data and lagged features.
        column (str): The target column for the autoregressive model.
        p (int): The order (number of lags) of the AR model.

    Returns:
        np.ndarray: An array of estimated coefficients, including the intercept term.
    """
    x = df[[f"{column}_lag{i}" for i in range(1, p + 1)]].to_numpy()
    y = df[column].to_numpy()

    x_with_intercept = np.column_stack((np.ones(x.shape[0]), x))

    coefficients = inv(x_with_intercept.T @ x_with_intercept) @ x_with_intercept.T @ y

    return coefficients


def _integrate_ar_coefficients(
    diff_coefficients: np.ndarray, *, differenced: bool
) -> pd.DataFrame:
    """Convert differenced AR model coefficients to integrated form.

    Args:
        diff_coefficients (np.ndarray): The coefficients from the differenced AR model.
        differenced (bool): Whether the model was fitted on differenced data.

    Returns:
        pd.DataFrame: A DataFrame with integrated coefficients and corresponding lags.
    """
    if not differenced:
        integrated_coeff = diff_coefficients
    else:
        integrated_coeff = np.zeros(len(diff_coefficients) + 1)
        integrated_coeff[0] = diff_coefficients[0]

        integrated_coeff[1] = 1 + diff_coefficients[1]

        for i in range(2, len(diff_coefficients)):
            integrated_coeff[i] = diff_coefficients[i - 1] - diff_coefficients[i]

        integrated_coeff[-1] = -diff_coefficients[-1]

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


def fit_ar_model(df: pd.DataFrame, column: str = "close_price", p: int = 1) -> dict:
    """Fit an autoregressive (AR) model of order p.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        column (str, optional): The target column to model. Defaults to "close_price".
        p (int, optional): The order of the AR model. Defaults to 1.

    Returns:
        dict: A dictionary containing:
            - "coefficients" (np.ndarray): Estimated coefficients of the AR(p) model.
            - "integrated_coefficients" (pd.DataFrame): Integrated coefficients.
            - "lag_order" (int): The order of the AR model (p).
            - "p_value" (float): The p-value from the stationarity test.
            - "differenced" (bool): Whether the series was differenced before fitting.

    """
    is_stationary, p_value, test_statistic_adf = _check_stationarity(df, column)
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
