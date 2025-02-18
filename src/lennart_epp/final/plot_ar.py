from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _predict_ar_from_coefficients(
    df: pd.DataFrame, integrated_coefficients: pd.DataFrame
) -> pd.Series:
    """Berechne die AR-Prognosen basierend auf den integrierten AR-Koeffizienten."""
    coeffs = integrated_coefficients["coefficient"].to_numpy()
    lag_order = len(coeffs) - 1

    predictions = [np.nan] * lag_order

    for i in range(lag_order, len(df)):
        ar_sum = coeffs[0]
        for lag in range(1, len(coeffs)):
            ar_sum += coeffs[lag] * df["close_price"].iloc[i - lag]
        predictions.append(ar_sum)

    return pd.Series(predictions, index=df.index)


def plot_top_ar_models(top_models: pd.DataFrame, df: pd.DataFrame, plot_path: str):
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
