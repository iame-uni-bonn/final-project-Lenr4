from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

msg = "Fehler beim Erstellen des PDF-Plots"


def _predict_ar_from_coefficients(
    df: pd.DataFrame, integrated_coefficients: pd.DataFrame
) -> pd.Series:
    coeffs = integrated_coefficients["coefficient"].to_numpy()
    lag_order = len(coeffs) - 1
    predictions = [np.nan] * lag_order
    for i in range(lag_order, len(df)):
        ar_sum = coeffs[0]
        for lag in range(1, len(coeffs)):
            ar_sum += coeffs[lag] * df["close_price"].iloc[i - lag]
        predictions.append(ar_sum)
    return pd.Series(predictions, index=df.index)


def plot_top_ar_models(
    top_models: pd.DataFrame,
    df: pd.DataFrame,
    plot_path: str,
    *,
    export_as_pdf: bool = False,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close_price"],
            mode="lines",
            name="Original Apple Stock Price",
            line={"color": "blue"},
        )
    )
    residual_traces = []
    colors = ["red", "green", "orange"]
    for i, (_, model) in enumerate(top_models.iterrows(), 1):
        predictions = _predict_ar_from_coefficients(
            df, model["integrated_coefficients"]
        )
        residuals = df["close_price"] - predictions
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=predictions,
                mode="lines",
                name=f"AR(p={model['p']}) Approximation",
                line={"color": colors[i - 1], "dash": "dash"},
            )
        )
        residual_traces.append(
            go.Scatter(
                x=df.index,
                y=residuals,
                mode="lines",
                name=f"Residuals AR(p={model['p']})",
                line={"color": colors[i - 1], "dash": "dot"},
            )
        )
    fig.update_layout(
        title="Apple Stock Price vs. Top 3 AR(p) Approximations",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Legend",
        template="plotly_white",
        height=500,
    )
    fig.update_xaxes(range=[df.index[-100], df.index[-1]])
    fig.update_yaxes(range=[150, 200])
    residual_fig = go.Figure(residual_traces)
    residual_fig.update_layout(
        title="Residuals of AR(p) Models",
        xaxis_title="Date",
        yaxis_title="Residuals",
        legend_title="Legend",
        template="plotly_white",
        height=300,
    )
    combined_html = (
        "<h1>Top 3 AR(p) Models vs. Original Data (Zoom on Last 100 Days, "
        "Y-axis 140-200)</h1>\n"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}\n"
        "<h2>Residuals of AR(p) Models</h2>\n"
        f"{residual_fig.to_html(full_html=False, include_plotlyjs='cdn')}"
    )

    with Path(plot_path).open("w", encoding="utf-8") as f:
        f.write(combined_html)
    if export_as_pdf:
        from plotly.subplots import make_subplots

        combined_fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Top AR(p) Models vs. Original Data",
                "Residuals of AR(p) Models",
            ),
            vertical_spacing=0.15,
        )
        for trace in fig.data:
            combined_fig.add_trace(trace, row=1, col=1)
        for trace in residual_fig.data:
            combined_fig.add_trace(trace, row=2, col=1)
        combined_fig.update_xaxes(range=[df.index[-100], df.index[-1]], row=1, col=1)
        combined_fig.update_yaxes(range=[150, 200], row=1, col=1)
        combined_fig.update_layout(
            height=800, title_text="Top AR(p) Models and Residuals"
        )
        pdf_path = Path(plot_path).with_suffix(".pdf")
        try:
            pio.write_image(
                combined_fig,
                str(pdf_path),
                format="pdf",
                width=1200,
                height=800,
                scale=2,
            )

        except OSError as e:
            raise RuntimeError(msg) from e
    return plot_path
