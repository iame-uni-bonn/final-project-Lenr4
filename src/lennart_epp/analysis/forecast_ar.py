import pandas as pd


def forecast_apple_prices(
    apple_df: pd.DataFrame, integrated_coefficients: pd.DataFrame
) -> pd.Series:
    last_date = apple_df.index[-1]
    # Forecast begins on 2023-01-03 or, if the last training date is later, on the
    # next business day.
    forecast_start = pd.to_datetime("2023-01-03")
    if last_date >= forecast_start:
        forecast_start = pd.bdate_range(start=last_date, periods=2)[1]
    forecast_index = pd.date_range(start=forecast_start, end="2023-12-31", freq="B")
    forecast_steps = len(forecast_index)

    # --- Step 2: Forecast using integrated coefficients ---

    coeffs = integrated_coefficients["coefficient"].to_numpy()
    p = len(coeffs) - 1
    # Use the last (p+1) price levels from the training data as the starting point.
    history = list(apple_df["close_price"].iloc[-(p + 1) :])
    forecasts = []

    for _ in range(forecast_steps):
        next_forecast = coeffs[0] + sum(
            coeffs[i] * history[-i] for i in range(1, p + 1)
        )
        forecasts.append(next_forecast)
        history.append(next_forecast)
        history = history[-(p + 1) :]

    return pd.Series(forecasts, index=forecast_index)
