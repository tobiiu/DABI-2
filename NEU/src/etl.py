import pandas as pd
from typing import Dict

def load_data(data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Loads all necessary datasets from specified paths.

    Args:
        data_paths (Dict[str, str]): A dictionary mapping dataset names to their file paths.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of loaded DataFrames.
    """
    orders = pd.read_parquet(data_paths['orders'])
    tips_public = pd.read_csv(data_paths['tips_public'])
    order_products = pd.read_csv(data_paths['order_products'])
    
    print("Data loaded successfully.")
    return {
        "orders": orders,
        "tips_public": tips_public,
        "order_products": order_products
    }

def clean_and_prepare_data(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Performs initial cleaning, type conversion, and preprocessing on the dataframes.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of raw dataframes.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cleaned dataframes.
    """
    orders = dataframes['orders'].copy()
    tips_public = dataframes['tips_public'].copy()
    order_products = dataframes['order_products'].copy()

    # Clean up column names and drop unnecessary ones
    if "Unnamed: 0" in tips_public.columns:
        tips_public = tips_public.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in order_products.columns:
        order_products = order_products.drop(columns=["Unnamed: 0"])
        
    # Standardize data types
    orders['order_id'] = orders['order_id'].astype('int64')
    orders['user_id'] = orders['user_id'].astype('int64')
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    tips_public['order_id'] = tips_public['order_id'].astype('int64')
    order_products['order_id'] = order_products['order_id'].astype('int64')
    order_products['product_id'] = order_products['product_id'].astype('int64')

    # Optimize memory usage with categorical types
    order_products['department'] = order_products['department'].astype('category')
    order_products['aisle'] = order_products['aisle'].astype('category')
    
    print("Data cleaning and preparation complete.")
    
    return {
        "orders": orders,
        "tips_public": tips_public,
        "order_products": order_products
    }