import pandas as pd

from lennart_epp.config import BLD, SRC
from lennart_epp.final.plot_ar import plot_top_ar_models
from lennart_epp.final.plot_forecast_ar import plot_forecast_ar

no_top_models_msg = "Keine Top-Modelle in der Auswertung gefunden."
file_creation_msg = "Datei konnte nicht erstellt werden."


def task_plot_top_ar_models(
    script=SRC / "final" / "plot_ar.py",
    evaluation_data=BLD / "models" / "ar_model_evaluation.pkl",
    produces=BLD / "plots" / "top_ar_models_plot.html",
):
    evaluation = pd.read_pickle(evaluation_data)

    top_models = pd.DataFrame(evaluation.get("top_models", []))

    if top_models.empty:
        raise ValueError(no_top_models_msg)

    plot_top_ar_models(
        top_models=top_models,
        df=pd.read_pickle(BLD / "data" / "cleaned_apple_data.pkl"),
        plot_path=str(produces),
        export_as_pdf=True,  # Änderung: jetzt wird auch ein PDF erzeugt
    )

    produces.parent.mkdir(parents=True, exist_ok=True)
    assert produces.exists(), f"Failed to produce plot at: {produces}"


def task_plot_forecast_ar(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    forecast=BLD / "forecasts" / "apple_2023_forecast.pkl",
    produces=BLD / "plots" / "apple_forecast_2023.html",
):
    plot_forecast_ar(
        data_path=data,
        forecast_path=forecast,
        output_path=str(produces),
        export_as_pdf=True,
    )

    produces.parent.mkdir(parents=True, exist_ok=True)

    assert produces.exists(), f"{file_creation_msg}: {produces}"
