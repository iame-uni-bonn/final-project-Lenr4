import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

missing_close_msg = "Close price column is missing in the original data."
empty_forecast_msg = "Forecast data is empty."
invalid_index_msg = "Forecast data does not have a DatetimeIndex."
msg_pdf = "Pdf_plot could not be created."


def plot_forecast_ar(
    data_path: str,
    forecast_path: str,
    output_path: str,
    *,
    export_as_pdf: bool = True,
):
    """Plot and save the original Apple stock price alongside the AR model forecast.

    Args:
        data_path (str): Path to the cleaned stock price data.
        forecast_path (str): Path to the AR model forecast data.
        output_path (str): Path to save the generated plot.
        export_as_pdf (bool, optional): If True, also exports the plot as a PDF.

    Returns:
        str: The path where the plot is saved.

    Raises:
        KeyError: If the 'close_price' column is missing in the input data.
        ValueError: If the forecast data is empty.
        TypeError: If the forecast index is not a DatetimeIndex.
        RuntimeError: If exporting to PDF fails.
    """
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
        title="Apple Stock Price: Original vs. AR-Multi-Step-Forecast",
        xaxis_title="Datum",
        yaxis_title="Kurs (USD)",
        legend_title="Legend",
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

    fig.write_html(output_path)

    if export_as_pdf:
        from pathlib import Path

        pdf_path = Path(output_path).with_suffix(".pdf")
        try:
            pio.write_image(
                fig, str(pdf_path), format="pdf", width=1200, height=600, scale=2
            )

        except OSError as e:
            raise RuntimeError(msg_pdf) from e
    return output_path
