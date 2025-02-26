import pandas as pd

from lennart_epp.config import BLD, SRC
from lennart_epp.final.plot_ar import plot_top_ar_models
from lennart_epp.final.plot_forecast_ar import plot_forecast_ar

no_top_models_msg = "Top-models not found."
file_creation_msg = "File could not be created."


def task_plot_top_ar_models(
    script=SRC / "final" / "plot_ar.py",
    evaluation_data=BLD / "models" / "ar_model_evaluation.pkl",
    produces=BLD / "plots" / "top_ar_models_plot.html",
):
    """Generate and save a plot of the top-performing AR models.

    Args:
        script (Path): Path to the script responsible for plotting.
        evaluation_data (Path): Path to the AR model evaluation data.
        produces (Path): Path where the output plot (HTML) will be stored.

    Returns:
        None: Ensures the top AR models are visualized and stored.

    Raises:
        ValueError: If no top models are available in the evaluation results.
    """
    evaluation = pd.read_pickle(evaluation_data)

    top_models = pd.DataFrame(evaluation.get("top_models", []))

    if top_models.empty:
        raise ValueError(no_top_models_msg)

    plot_top_ar_models(
        top_models=top_models,
        df=pd.read_pickle(BLD / "data" / "cleaned_apple_data.pkl"),
        plot_path=str(produces),
        export_as_pdf=True,
    )

    produces.parent.mkdir(parents=True, exist_ok=True)
    assert produces.exists(), f"Failed to produce plot at: {produces}"


def task_plot_forecast_ar(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    forecast=BLD / "forecasts" / "multistep_forecast.pkl",
    produces=BLD / "plots" / "multistep_forecast.html",
):
    """Generate and save a visualization of the AR model multi-step forecast.

    Args:
        data (Path): Path to the cleaned stock price data.
        forecast (Path): Path to the multi-step forecast data.
        produces (Path): Path to store the generated forecast plot.

    Returns:
        None: Ensures the forecast visualization is generated and saved.

    Raises:
        AssertionError: If the output plot file is not created successfully.
    """
    plot_forecast_ar(
        data_path=data,
        forecast_path=forecast,
        output_path=str(produces),
        export_as_pdf=True,
    )

    produces.parent.mkdir(parents=True, exist_ok=True)

    assert produces.exists(), f"{file_creation_msg}: {produces}"
