import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

missing_close_msg = "Close price column is missing in the original data."
empty_forecast_msg = "Forecast data is empty."
invalid_index_msg = "Forecast data does not have a DatetimeIndex."


def plot_forecast_ar(
    data_path: str,
    forecast_path: str,
    output_path: str,
    *,
    export_as_pdf: bool = False,
):
    df = pd.read_pickle(data_path)
    forecast = pd.read_pickle(forecast_path)

    if "close_price" not in df.columns:
        raise KeyError(missing_close_msg)

    original = df["close_price"].loc["2022-07-17":"2022-09-10"]

    if forecast.empty:
        raise ValueError(empty_forecast_msg)
    if not isinstance(forecast.index, pd.DatetimeIndex):
        raise TypeError(invalid_index_msg)

    forecast = forecast.loc["2022-07-17":"2022-09-10"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode="lines",
            name="Originaldaten (close_price)",
            line={"color": "blue", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            name="Forecast (AR-Modell Step=10)",
            line={"color": "orange", "width": 2, "dash": "dash"},
        )
    )

    fig.update_layout(
        title="Apple Aktienkurs: Original vs. AR-Forecast (2023)",
        xaxis_title="Datum",
        yaxis_title="Kurs (USD)",
        legend_title="Legende",
        template="plotly_white",
        autosize=True,
        yaxis={"range": [143, 180]},
    )

    fig.add_shape(
        {
            "type": "line",
            "x0": "2022-08-18",
            "y0": 143,
            "x1": "2022-08-18",
            "y1": 180,
            "line": {
                "color": "red",
                "width": 2,
                "dash": "solid",
            },
        }
    )

    if export_as_pdf:
        pio.write_image(fig, output_path, format="pdf", width=1200, height=600, scale=2)
    else:
        fig.write_html(output_path)
