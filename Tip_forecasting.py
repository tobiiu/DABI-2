import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def calculate_hourly_tip_shares(orders_df, tips_df):
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    df = pd.merge(orders_df, tips_df, on='order_id', how='left')
    df['tip'] = df['tip'].fillna(0)
    df = df.set_index('order_date')

    hourly_orders = df['tip'].resample('h').count()
    hourly_tip_share = df['tip'].resample('h').mean()

    return pd.DataFrame({
        'tip_share': hourly_tip_share,
        'num_orders': hourly_orders
    })


def trim_stable_time_period(df, min_orders=20):
    valid = df['num_orders'] >= min_orders
    if not valid.any():
        raise ValueError("No time periods with enough orders found.")
    start = df.index[valid.argmax()]
    end = df.index[::-1][valid[::-1].argmax()]
    return df.loc[start:end]


def remove_trend_and_seasonality(df):
    df = df.copy()
    df['d_tip_share'] = df['tip_share'].diff().dropna()
    return df


def sarimax_forecast(df, steps=24):
    series = df['tip_share'].fillna(method='ffill')

    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    forecast = result.forecast(steps=steps)

    forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=steps, freq='h')
    forecast_df = pd.DataFrame({
        'tip_share': forecast,
        'num_orders': np.nan,
        'd_tip_share': np.nan,
        'is_forecast': True
    }, index=forecast_index)

    df['is_forecast'] = False
    combined = pd.concat([df, forecast_df])
    return combined


def plot_time_series(before_df, after_df):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(before_df['tip_share'], label='Original')
    plt.title("Original Time Series (Trimmed)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(after_df['tip_share'], label='With Forecast', color='red')
    plt.title("Time Series with Forecast")
    plt.legend()

    plt.tight_layout()
    plt.show()


def process_tip_time_series(orders_df, tips_df, min_orders=20, forecast_hours=24):
    hourly_df = calculate_hourly_tip_shares(orders_df, tips_df)
    trimmed_df = trim_stable_time_period(hourly_df, min_orders)
    detrended_df = remove_trend_and_seasonality(trimmed_df)
    forecast_df = sarimax_forecast(detrended_df, steps=forecast_hours)
    plot_time_series(detrended_df, forecast_df)
    return forecast_df


# Optional: Forecast visualization with `is_forecast` column

def plot_forecast_with_split(df):
    """
    Plots the tip share time series, distinguishing between actual data and forecasted values.
    Requires a boolean 'is_forecast' column.
    """
    df = df.copy().sort_index()

    if 'is_forecast' not in df.columns:
        raise ValueError("Missing 'is_forecast' column. Please use sarimax_forecast() properly.")

    df_actual = df[df['is_forecast'] == False]
    df_forecast = df[df['is_forecast'] == True]

    plt.figure(figsize=(12, 4))
    plt.plot(df_actual.index, df_actual['tip_share'], label='Actual', color='blue')
    plt.plot(df_forecast.index, df_forecast['tip_share'], label='Forecast', color='red')

    if not df_forecast.empty:
        plt.axvline(x=df_forecast.index[0], color='gray', linestyle='--', label='Forecast Starts')

    plt.title("Hourly Tip Share with Forecast")
    plt.xlabel("Time")
    plt.ylabel("Tip Share")
    plt.legend()
    plt.tight_layout()
    plt.show()
