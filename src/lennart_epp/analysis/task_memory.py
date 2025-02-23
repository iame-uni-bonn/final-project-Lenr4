import pandas as pd

from lennart_epp.analysis.memory import (
    check_stat_diff_close,
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
