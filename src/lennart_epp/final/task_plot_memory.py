import pandas as pd

from lennart_epp.config import BLD, SRC
from lennart_epp.final.plot_memory import plot_acf


def task_plot_acf(
    script=SRC / "final" / "plot_memory.py",
    acf_data=BLD / "memory" / "acf.pkl",
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=(
        BLD / "plots" / "acf_plot.html",
        BLD / "plots" / "acf_plot.pdf",
    ),
):
    """Task to generate and save the ACF plot for the differenced close price.

    Args:
        script (Path): Path to the script that generates the ACF plot.
        acf_data (Path): Path to the computed ACF values stored in memory.
        data (Path): Path to the original cleaned Apple stock data.
        produces (tuple[Path, Path]): Paths to the output HTML and PDF plots.

    Returns:
        None: Ensures the plots are successfully generated.
    """
    acf_results = pd.read_pickle(acf_data)

    df = pd.read_pickle(data)
    n_obs = df["close_price"].diff().dropna().shape[0]

    plot_acf(acf_results, produces[0], produces[1], n_obs)

    for file in produces:
        assert file.exists(), f"Failed to produce {file}"
