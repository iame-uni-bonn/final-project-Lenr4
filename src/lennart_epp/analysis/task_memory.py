import pandas as pd

from lennart_epp.analysis.memory import (
    check_stat_diff_close,
    compute_acf,
    compute_hurst_exponent,
    write_hurst_result_to_tex,
    write_stationarity_results,
)
from lennart_epp.config import BLD


def task_check_stat_diff_close(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "memory" / "diff_close_stat_test.tex",
):
    """Task to check if the differenced 'close_price' is stationary, and saves as .tex.

    Args:
        data (Path): Path to the cleaned Apple stock data (Pickle file).
        produces (Path): Path to the output LaTeX file.

    Returns:
        None: Saves results to a .tex file.
    """
    df = pd.read_pickle(data)

    results = check_stat_diff_close(df, column="close_price")

    write_stationarity_results(results, produces)

    assert produces.exists(), f" Failed to produce {produces}"


def task_compute_acf(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "memory" / "acf.pkl",
):
    """Task to compute and save ACF values manually for the differenced close price.

    Args:
        data (Path): Path to the cleaned Apple stock data (Pickle file).
        produces (Path): Path to output pickle file where the ACF results are stored.

    Returns:
        None: Saves ACF values to a .pkl file.
    """
    df = pd.read_pickle(data)

    results = compute_acf(df, column="close_price")

    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, produces)

    assert produces.exists(), f"Failed to produce {produces}"


def task_hurst_exponent(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "memory" / "hurst_exponent.tex",
):
    """Task to compute the Hurst exponent and store results as LaTeX file.

    Args:
        data (Path): Path to the cleaned Apple stock data.
        produces (Path): Path to the output LaTeX file.

    Returns:
        None: Saves results as a LaTeX file.
    """
    df = pd.read_pickle(data)
    df["diff_close_price"] = df["close_price"].diff().dropna()

    results = compute_hurst_exponent(df, column="diff_close_price")

    write_hurst_result_to_tex(results, produces)

    assert produces.exists(), f"Failed to produce {produces}"
