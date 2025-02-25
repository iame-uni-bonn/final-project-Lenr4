import numpy as np
import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model
from lennart_epp.analysis.forecast_ar import forecast_ar_multi_step
from lennart_epp.config import BLD

missing_close_price_msg = "Column 'close_price' missing!"
too_few_train_msg = "Not enough training data."
type_forecast_msg = "Forecast has to be 'pd.Series'."
multi_forecast_msg = "Forecast contains multi-dimensional values."


def task_forecast_ar(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "forecasts" / "multistep_forecast.pkl",
    lags=50,
):
    """Generate multi-step forecasts using an AR model.

    Args:
        data (Path): Path to the cleaned Apple stock data.
        produces (Path): Path to store the multi-step forecast as a pickle file.
        lags (int, optional): Number of lags to use for the AR model.

    Raises:
        KeyError: If the required "close_price" column is missing.
        ValueError: If there are insufficient training data.
        TypeError: If the forecast is not a Pandas Series.
        ValueError: If the forecast contains multi-dimensional values.

    Returns:
        None: Saves the forecast to specified output files.
    """
    df = pd.read_pickle(data)
    if "close_price" not in df.columns:
        raise KeyError(missing_close_price_msg)

    apple_prices = df["close_price"].asfreq("B", method="ffill")
    train_data = apple_prices.loc[apple_prices.index < "2022-8-20"]
    if len(train_data) < lags:
        raise ValueError(too_few_train_msg)

    ar_result = fit_ar_model(train_data.to_frame(), column="close_price", p=lags)
    integrated_coefficients = ar_result["integrated_coefficients"]

    forecast = forecast_ar_multi_step(
        df,
        integrated_coefficients=integrated_coefficients,
        forecast_steps=lags,
    )

    if not isinstance(forecast, pd.Series):
        raise TypeError(type_forecast_msg)
    if forecast.apply(lambda x: isinstance(x, list | np.ndarray)).any():
        raise ValueError(multi_forecast_msg)

    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(forecast, produces)

    html_output = produces.with_suffix(".html")
    forecast.to_frame(name="Forecasted Close Price").to_html(html_output, index=True)
