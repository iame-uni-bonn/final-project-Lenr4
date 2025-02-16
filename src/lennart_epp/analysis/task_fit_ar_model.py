"""Task for fitting AR(p) model and saving results."""

import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model
from lennart_epp.config import BLD, SRC


def task_fit_ar_model(
    script=SRC / "analysis" / "fit_ar_model.py",
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "models" / "ar_model_output.pkl",
    p=15,
):
    """Fit AR(p) model and save coefficients, integrated coefficients, and metadata."""
    df = pd.read_pickle(data)

    model_results = fit_ar_model(df, column="close_price", p=p)

    coeff_df = pd.DataFrame(
        {
            "coefficient": model_results["coefficients"],
            "lag": [
                f"Lag {i}" if i > 0 else "Intercept"
                for i in range(len(model_results["coefficients"]))
            ],
        }
    )

    integrated_coeff_df = model_results["integrated_coefficients"]

    metadata_df = pd.DataFrame(
        {
            "p_value": [model_results["p_value"]],
            "differenced": [model_results["differenced"]],
            "lag_order": [model_results["lag_order"]],
        }
    )

    results = {
        "coefficients": coeff_df,
        "integrated_coefficients": integrated_coeff_df,
        "metadata": metadata_df,
    }

    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, produces)

    assert produces.exists(), f"❌ Failed to produce {produces}"
