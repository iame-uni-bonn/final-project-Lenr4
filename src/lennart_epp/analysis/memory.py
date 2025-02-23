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
