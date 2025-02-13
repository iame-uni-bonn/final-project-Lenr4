"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from lennart_epp.config import BLD, SRC, TEMPLATE_GROUPS
from lennart_epp.final.plot_template import plot_regression_by_age

# Only keep Python-based tasks
for group in TEMPLATE_GROUPS:

    @pytask.task(id=f"{group}_python")  # Removed language argument
    def task_plot_results_by_age(
        script=SRC / "final" / "plot_template.py",
        group=group,
        predictions_path=BLD / "predictions" / f"{group}.pickle",  # Use only .pickle
        data_path=BLD / "data" / "stats4schools_smoking.pickle",  # Use only .pickle
        produces=BLD / "figures" / f"smoking_by_{group}_using_python.png",
    ):
        """Plot the regression results by age."""
        data = pd.read_pickle(data_path)  # Load only Python-compatible data
        predictions = pd.read_pickle(predictions_path)
        fig = plot_regression_by_age(data, predictions, group)
        fig.write_image(produces)


def task_create_results_table(
    script=SRC / "final" / "plot_template.py",
    model_path=BLD / "estimation_results" / "baseline.pickle",
    produces=BLD / "tables" / "estimation_results.tex",
):
    """Store a table in LaTeX format with the estimation results)."""
    model = pd.read_pickle(model_path)
    table = model.summary().as_latex()
    with produces.open("w") as f:
        f.writelines(table)
