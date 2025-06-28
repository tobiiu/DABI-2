import os
from prefect import task, flow
from typing import Dict, Any
import pandas as pd

# Import functions from your modules
from src.etl import load_data, clean_and_prepare_data
from src.features import combine_all_features
from src.analysis import plot_tipping_periodicity, plot_tipping_trend
from src.modeling import train_and_evaluate_model, make_predictions
from src.forecasting import process_tip_time_series, plot_forecast_with_split

# Define file paths
DATA_PATHS = {
    'orders': 'data/orders.parquet',
    'tips_public': 'data/tips_public.csv',
    'order_products': 'data/order_products_denormalized.csv'
}

@task(name="Load and Clean Data")
def load_and_clean_task(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Prefect task to load and clean the initial datasets."""
    raw_data = load_data(paths)
    cleaned_data = clean_and_prepare_data(raw_data)
    return cleaned_data

@task(name="Time Series Analysis")
def analysis_task(cleaned_data: Dict[str, pd.DataFrame]):
    """Prefect task to perform and visualize time series analysis."""
    print("\n--- Running Periodicity and Trend Analysis ---")
    plot_tipping_periodicity(cleaned_data['orders'], cleaned_data['tips_public'])
    plot_tipping_trend(cleaned_data['orders'], cleaned_data['tips_public'])

@task(name="SARIMAX Forecasting")
def forecasting_task(cleaned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prefect task for advanced SARIMAX time series forecasting. Returns the forecast."""
    print("\n--- Running SARIMAX Forecasting ---")
    forecast_df = process_tip_time_series(
        orders_df=cleaned_data['orders'],
        tips_df=cleaned_data['tips_public'],
        min_orders=30,
        forecast_hours=24*7
    )
    plot_forecast_with_split(forecast_df)
    forecast_df.to_csv('output/sarimax_forecast.csv')
    print("SARIMAX forecast saved to output/sarimax_forecast.csv")
    return forecast_df

@task(name="Feature Engineering")
def feature_engineering_task(cleaned_data: Dict[str, pd.DataFrame], forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Prefect task to generate all features, including lagged and forecast features."""
    print("\n--- Running Feature Engineering ---")
    all_features_df = combine_all_features(
        orders=cleaned_data['orders'],
        order_products_denormalized=cleaned_data['order_products'],
        tips_public=cleaned_data['tips_public'],
        forecast_df=forecast_df # Pass the forecast here
    )
    all_features_df.to_csv('output/all_features.csv', index=False)
    print("Engineered features saved to output/all_features.csv")
    return all_features_df

@task(name="Train Model and Predict")
def train_predict_task(features_df: pd.DataFrame, force_retrain: bool):
    """Prefect task to train the model, evaluate it, and make predictions."""
    print("\n--- Training Model and Making Predictions ---")
    model = train_and_evaluate_model(features_df, force_retrain=force_retrain)
    predictions = make_predictions(model, features_df)
    print(f"\nGenerated {len(predictions)} predictions.")

@flow(name="Tip Prediction Pipeline")
def tip_prediction_flow(force_retrain: bool = False):
    """
    The main Prefect flow to run the entire data science pipeline.

    Args:
        force_retrain (bool): Set to True to force the model to retrain.
                              Defaults to False, which loads a saved model if available.
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    cleaned_data = load_and_clean_task(DATA_PATHS)
    
    analysis_task.submit(cleaned_data) # Run analysis in the background
    
    # The forecast must complete before feature engineering
    forecast_result = forecasting_task.submit(cleaned_data)
    
    features_df = feature_engineering_task(
        cleaned_data=cleaned_data,
        forecast_df=forecast_result.result() # Get the result from the forecast task
    )
    
    train_predict_task(features_df, force_retrain=force_retrain)

if __name__ == "__main__":
    # Example of how to run the flow
    # To force retraining: tip_prediction_flow(force_retrain=True)
    tip_prediction_flow(force_retrain=False)