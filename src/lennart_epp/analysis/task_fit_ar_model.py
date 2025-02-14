"""Tasks for fitting the AR(p) model on Apple stock data."""

import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model
from lennart_epp.config import BLD, SRC


def task_fit_ar_model(
    script=SRC / "analysis" / "fit_ar_model.py",
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "models" / "ar_model_output.pkl",
    p=5,
):
    # Load the cleaned stock data
    df = pd.read_pickle(data)

    # Fit the AR(p) model
    model_results = fit_ar_model(df, column="close_price", p=p)

    # Prepare and save model results
    produces.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame for coefficients only
    coeff_df = pd.DataFrame(
        {
            "coefficient": model_results["coefficients"],
            "lag": [
                f"Lag {i}" if i > 0 else "Intercept"
                for i in range(len(model_results["coefficients"]))
            ],
        }
    )

    # Create a metadata DataFrame with p-value and differencing info
    metadata_df = pd.DataFrame(
        {
            "p_value": [model_results["p_value"]],
            "differenced": [model_results["differenced"]],
            "lag_order": [model_results["lag_order"]],
        }
    )

    # Combine into a single dictionary and save
    result = {"coefficients": coeff_df, "metadata": metadata_df}

    pd.to_pickle(result, produces)

    assert produces.exists(), f"Failed to produce {produces}"
