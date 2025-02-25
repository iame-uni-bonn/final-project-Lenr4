import pandas as pd


def forecast_ar_multi_step(
    df: pd.DataFrame, integrated_coefficients: pd.DataFrame, forecast_steps: int
) -> pd.Series:
    """Generate multi-step forecasts using an AR model.

    This function uses the integrated AR model coefficients to iteratively
    forecast future values.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        integrated_coefficients (pd.DataFrame): DataFrame of integrated AR coefficients.
        forecast_steps (int): Number of future steps to forecast.

    Returns:
        pd.Series: A time series of forecasted values indexed dates.

    """
    coeffs = integrated_coefficients["coefficient"].to_numpy()
    lag_order = len(coeffs) - 1

    history = df["close_price"].iloc[-lag_order:].tolist()
    forecasts = []

    for _ in range(forecast_steps):
        forecast_val = coeffs[0]
        for lag in range(1, len(coeffs)):
            forecast_val += coeffs[lag] * history[-lag]
        forecasts.append(forecast_val)

        history.append(forecast_val)
        history.pop(0)

    last_date = df.index[-345]
    forecast_index = pd.date_range(
        start=last_date, periods=forecast_steps + 1, freq="D"
    )[1:]

    return pd.Series(forecasts, index=forecast_index)
