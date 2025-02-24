import numpy as np
import plotly.graph_objects as go


def plot_acf(acf_data: dict, save_path_html: str, save_path_pdf: str, n_obs: int):
    """Plot the manually computed ACF using Plotly with confidence bands.

    Args:
        acf_data (dict): Dictionary containing "acf" values and "lags".
        save_path_html (str): Path to save the interactive HTML plot.
        save_path_pdf (str): Path to save the static PDF plot.
        n_obs (int): Number of observations in the time series (for confidence bands).

    Returns:
        None: Saves plots in the specified paths.
    """
    max_lags = 200
    lags = acf_data["lags"][:max_lags]
    acf_values = acf_data["acf"][:max_lags]

    confidence_band = 1.96 / np.sqrt(n_obs)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=lags, y=acf_values, marker={"color": "rgba(0, 0, 255, 0.8)"}, name="ACF"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[confidence_band] * len(lags),
            mode="lines",
            line={"color": "red", "dash": "dash"},
            name="95% Confidence Interval",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[-confidence_band] * len(lags),
            mode="lines",
            line={"color": "red", "dash": "dash"},
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Autocorrelation Function (ACF) with Confidence Bands",
        xaxis_title="Lag",
        yaxis_title="ACF Value",
        template="plotly_white",
        yaxis={"range": [-0.15, 0.3]},
        plot_bgcolor="whitesmoke",
        xaxis={"gridcolor": "lightgray"},
    )

    fig.write_html(save_path_html)
    fig.write_image(save_path_pdf)
