from pathlib import Path

import nolds
import numpy as np
import pandas as pd

from lennart_epp.analysis.fit_ar_model import _check_stationarity, _difference_series

error_msg = "Column '{column}' not found in dataframe."


def check_stat_diff_close(df: pd.DataFrame, column: str = "close_price") -> dict:
    """Check if the first-differenced close_price is stationary.

    Args:
        df (pd.DataFrame): The dataframe containing the time series.
        column (str, optional): Column to test for stationarity.

    Returns:
        dict: A dictionary containing the ADF test results.
    """
    if column not in df.columns:
        raise ValueError(error_msg)

    df_diff = _difference_series(df, column)

    diff_column = f"diff_{column}"

    is_stationary, p_value, test_statistic_adf = _check_stationarity(
        df_diff, diff_column
    )

    return {
        "ADF Test Statistic": test_statistic_adf,
        "P-Value": p_value,
        "Is Stationary": is_stationary,
    }


def write_stationarity_results(results: dict, file_path):
    """Generate and save a LaTeX table for Augmented Dickey-Fuller (ADF) test results.

    Args:
        results (dict): A dictionary containing the ADF test results with keys:
            - "ADF Test Statistic" (float or None): The test statistic value.
            - "P-Value" (float): The p-value from the ADF test.
            - "Is Stationary" (bool): Whether the series is stationary.
        file_path (Path): The file path where the LaTeX table will be saved.

    Returns:
        None: The function writes the LaTeX table to the specified file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    test_stat = results["ADF Test Statistic"]
    test_stat_str = f"{test_stat:.4f}" if test_stat is not None else "Not Available"

    conclusion_text = (
        "The differenced series is likely stationary."
        if results["Is Stationary"]
        else "The differenced series may not be stationary."
    )

    latex_content = f"""
    \\begin{{table}}[H]
        \\centering
        \\caption{{Results of the Augmented Dickey-Fuller (ADF) Test}}
        \\label{{tab:stationarity_test}}
        \\begin{{tabular}}{{l c}}
            \\toprule
            \\textbf{{Test Statistic}} & \\textbf{{Value}} \\\\
            \\midrule
            ADF Test Statistic & {test_stat_str} \\\\
            P-Value & {results["P-Value"]:.4f} \\\\
            Conclusion & {conclusion_text} \\\\
            \\bottomrule
        \\end{{tabular}}
    \\end{{table}}
    """

    with file_path.open("w", encoding="utf-8") as f:
        f.write(latex_content.strip())


def _compute_mean(series: np.ndarray) -> float:
    """Compute the mean of a given time series.

    Args:
        series (np.ndarray): The time series data as a NumPy array.

    Returns:
        float: The mean of the time series.
    """
    return np.mean(series)


def _compute_variance(series: np.ndarray, mean_series: float) -> float:
    """Compute the variance of a time series.

    Args:
        series (np.ndarray): The time series data as a NumPy array.
        mean_series (float): The precomputed mean of the series.

    Returns:
        float: The variance of the time series.
    """
    return np.sum((series - mean_series) ** 2)


def _compute_autocovariance(series: np.ndarray, mean_series: float, lag: int) -> float:
    """Compute the autocovariance for a given lag in a time series.

    Args:
        series (np.ndarray): The time series data as a NumPy array.
        mean_series (float): The precomputed mean of the series.
        lag (int): The lag at which to compute the autocovariance.

    Returns:
        float: The autocovariance value for the specified lag.
    """
    n = len(series)
    return np.sum((series[lag:] - mean_series) * (series[: n - lag] - mean_series))


def compute_acf(
    df: pd.DataFrame, column: str = "close_price", lags: int = 1000
) -> dict:
    """Compute the ACF manually for the first-differenced time series.

    Args:
        df (pd.DataFrame): The dataframe containing the time series.
        column (str, optional): Column to analyze. Defaults to "close_price".
        lags (int, optional): Number of lags for ACF.

    Returns:
        dict: A dictionary containing ACF values and corresponding lags.
    """
    df_diff = _difference_series(df, column)
    series = df_diff[f"diff_{column}"].dropna().to_numpy()

    lags = min(len(series) - 1, lags)
    mean_series = _compute_mean(series)
    variance = _compute_variance(series, mean_series)

    acf_values = []
    for lag in range(lags + 1):
        autocovariance = _compute_autocovariance(series, mean_series, lag)
        acf_values.append(autocovariance / variance)

    return {"acf": np.array(acf_values), "lags": np.arange(len(acf_values))}


def compute_hurst_exponent(df: pd.DataFrame, column: str = "close_price") -> dict:
    """Compute the Hurst exponent to assess long-memory effects.

    Args:
        df (pd.DataFrame): The dataframe containing the time series.
        column (str, optional): Column to analyze. Defaults to "close_price".

    Returns:
        dict: A dictionary containing the computed Hurst exponent.
    """
    series = df[column].dropna().to_numpy()
    hurst_value = nolds.hurst_rs(series)

    return {"Hurst Exponent": hurst_value}


def write_hurst_result_to_tex(results: dict, file_path: Path):
    """Write the computed Hurst exponent results to a LaTeX file.

    Args:
        results (dict): Dictionary containing the Hurst exponent.
        file_path (Path): Path where the LaTeX file will be saved.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    hurst_value = results["Hurst Exponent"]

    latex_content = f"""
    \\begin{{table}}[H]
        \\centering
        \\caption{{Hurst Exponent Statistics}}
        \\label{{tab:hurst_exponent}}
        \\begin{{tabular}}{{l c}}
            \\toprule
            \\textbf{{Metric}} & \\textbf{{Value}} \\\\
            \\midrule
            Hurst Exponent & {hurst_value:.4f} \\\\
            \\bottomrule
        \\end{{tabular}}
    \\end{{table}}
    """

    file_path.write_text(latex_content.strip(), encoding="utf-8")
