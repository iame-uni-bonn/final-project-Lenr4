import numpy as np
import pandas as pd

from lennart_epp.analysis.fit_ar_model import fit_ar_model
from lennart_epp.analysis.forecast_ar import forecast_apple_prices
from lennart_epp.config import BLD

# Konstanten für Fehlermeldungen
missing_close_price_msg = "Spalte 'close_price' fehlt!"
too_few_train_msg = "Zu wenige Trainingsdaten."
type_forecast_msg = "Forecast muss 'pd.Series' sein."
multi_forecast_msg = "Forecast enthält mehrdimensionale Werte."


def task_forecast_ar(
    data=BLD / "data" / "cleaned_apple_data.pkl",
    produces=BLD / "forecasts" / "apple_2023_forecast.pkl",
    lags=5,
):
    df = pd.read_pickle(data)
    if "close_price" not in df.columns:
        raise KeyError(missing_close_price_msg)

    apple_prices = df["close_price"].asfreq("B", method="ffill")
    train_data = apple_prices.loc[apple_prices.index < "2023-01-03"]
    if len(train_data) < lags:
        raise ValueError(too_few_train_msg)

    # AR-Modell fitten, um die integrierten Koeffizienten zu erhalten
    ar_result = fit_ar_model(train_data.to_frame(), column="close_price", p=lags)
    integrated_coefficients = ar_result["integrated_coefficients"]

    # Forecast mit den integrierten Koeffizienten berechnen
    forecast = forecast_apple_prices(
        apple_df=train_data.to_frame(),
        integrated_coefficients=integrated_coefficients,
    )

    # Validierung before dem Speichern
    if not isinstance(forecast, pd.Series):
        raise TypeError(type_forecast_msg)
    if forecast.apply(lambda x: isinstance(x, list | np.ndarray)).any():
        raise ValueError(multi_forecast_msg)

    produces.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(forecast, produces)

    html_output = produces.with_suffix(".html")
    forecast.to_frame(name="Forecasted Close Price").to_html(html_output, index=True)
