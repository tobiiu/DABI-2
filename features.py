import numpy as np
import pandas as pd
from IPython.display import display

############ HELPER FUNCTIONS ##########################

def merge_with_orders(df, orders, columns=['order_id', 'user_id']):
    """Merge a DataFrame with orders to include user_id and/or other columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing order_id.
        orders (pd.DataFrame): Orders dataset.
        columns (list): Columns from orders to include.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    return df.merge(orders[columns], on='order_id', how='left')

def validate_dataframe(df, required_columns):
    """Check if DataFrame has required columns and is not empty.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Raises:
        ValueError: If validation fails.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def cyclic_transform(value, max_value):
    """Apply sine-cosine transformation to a cyclical value.
    
    Args:
        value (pd.Series): Values to transform (e.g., hour or month).
        max_value (float): Maximum value of the cycle (e.g., 24 for hours, 12 for months).
    
    Returns:
        tuple: (sin_values, cos_values) as pd.Series.
    """
    radians = 2 * np.pi * value / max_value
    return np.sin(radians), np.cos(radians)

def compute_median_time_diff(orders):
    """Compute median time difference between consecutive orders across all users.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        float: Median time difference in hours.
    """
    sorted_orders = orders[['user_id', 'order_date']].sort_values(['user_id', 'order_date'])
    time_diffs = sorted_orders.groupby('user_id')['order_date'].diff().dt.total_seconds() / 3600
    return time_diffs.median() if not time_diffs.empty else 24.0  # Default to 24 hours if empty

def downcast_dtypes(df, exclude_columns=['order_id', 'user_id', 'product_id']):
    """Downcast numeric columns to reduce memory usage, excluding specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame to downcast.
        exclude_columns (list): Columns to exclude from downcasting.
    
    Returns:
        pd.DataFrame: Downcasted DataFrame.
    """
    for col in df.select_dtypes(include=['int64']).columns:
        if col not in exclude_columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

####################### User-Level Features ######################
# Each feature returns a DataFrame with columns: user_id, feature_name.

def add_feature_alcohol_count(order_products_denormalized, orders):
    """Count how many alcohol products each user has purchased.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_alcohol_purchase_count].
    """
    alcohol_df = order_products_denormalized[order_products_denormalized['department'] == 'alcohol']
    if alcohol_df.empty:
        return pd.DataFrame({'user_id': orders['user_id'].unique(), 'user_alcohol_purchase_count': 0})
    
    alcohol_with_users = merge_with_orders(alcohol_df, orders)
    alcohol_counts = alcohol_with_users.groupby('user_id').size().reset_index(name='user_alcohol_purchase_count')
    
    all_users = pd.DataFrame({'user_id': orders['user_id'].unique()})
    result = all_users.merge(alcohol_counts, on='user_id', how='left').fillna({'user_alcohol_purchase_count': 0})
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_alcohol_purchase_count'])
    return result

def total_products_per_user(order_products_denormalized, orders):
    """Count total products purchased by each user.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_total_purchase_count].
    """
    merged = merge_with_orders(order_products_denormalized, orders)
    total_products = merged.groupby('user_id')['product_id'].count().reset_index(name='user_total_purchase_count')
    total_products = downcast_dtypes(total_products)
    
    validate_dataframe(total_products, ['user_id', 'user_total_purchase_count'])
    return total_products

def total_unique_products_per_user(order_products_denormalized, orders):
    """Count unique products purchased by each user.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_unique_product_count].
    """
    merged = merge_with_orders(order_products_denormalized, orders)
    unique_products = merged.groupby('user_id')['product_id'].nunique().reset_index(name='user_unique_product_count')
    unique_products = downcast_dtypes(unique_products)
    
    validate_dataframe(unique_products, ['user_id', 'user_unique_product_count'])
    return unique_products

def unique_to_total_product_ratio_per_user(order_products_denormalized, orders):
    """Calculate ratio of unique to total products purchased by each user.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_unique_to_total_ratio].
    """
    total = total_products_per_user(order_products_denormalized, orders)
    unique = total_unique_products_per_user(order_products_denormalized, orders)
    merged = total.merge(unique, on='user_id')
    merged['user_unique_to_total_ratio'] = merged['user_unique_product_count'] / merged['user_total_purchase_count']
    merged = downcast_dtypes(merged)
    
    validate_dataframe(merged, ['user_id', 'user_unique_to_total_ratio'])
    return merged[['user_id', 'user_unique_to_total_ratio']]

def most_frequent_purchase_hour(orders):
    """Identify the most frequent hour of day for each user's orders.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_frequent_purchase_hour].
    """
    orders_with_hour = orders[['user_id', 'order_date']].copy()
    orders_with_hour['hour'] = orders_with_hour['order_date'].dt.hour
    
    hour_counts = orders_with_hour.groupby(['user_id', 'hour']).size().reset_index(name='count')
    idx = hour_counts.groupby('user_id')['count'].idxmax()
    result = hour_counts.loc[idx, ['user_id', 'hour']].rename(columns={'hour': 'user_frequent_purchase_hour'})
    
    all_users = pd.DataFrame({'user_id': orders['user_id'].unique()})
    result = all_users.merge(result, on='user_id', how='left').fillna({'user_frequent_purchase_hour': 12})  # Default to noon
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_frequent_purchase_hour'])
    return result

def most_frequent_purchase_dow(orders):
    """Identify the most frequent day of week for each user's orders (0=Monday, 6=Sunday).
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_frequent_purchase_dow].
    """
    orders_with_dow = orders[['user_id', 'order_date']].copy()
    orders_with_dow['dow'] = orders_with_dow['order_date'].dt.dayofweek
    
    dow_counts = orders_with_dow.groupby(['user_id', 'dow']).size().reset_index(name='count')
    idx = dow_counts.groupby('user_id')['count'].idxmax()
    result = dow_counts.loc[idx, ['user_id', 'dow']].rename(columns={'dow': 'user_frequent_purchase_dow'})
    
    all_users = pd.DataFrame({'user_id': orders['user_id'].unique()})
    result = all_users.merge(result, on='user_id', how='left').fillna({'user_frequent_purchase_dow': 0})  # Default to Monday
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_frequent_purchase_dow'])
    return result

def avg_time_between_orders(orders):
    """Calculate average time between consecutive orders for each user in hours.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_avg_order_interval_hours].
    """
    sorted_orders = orders[['user_id', 'order_date']].sort_values(['user_id', 'order_date'])
    time_diffs = sorted_orders.groupby('user_id')['order_date'].diff().dt.total_seconds() / 3600
    
    avg_diffs = time_diffs.groupby(sorted_orders['user_id']).mean().reset_index(name='user_avg_order_interval_hours')
    all_users = pd.DataFrame({'user_id': orders['user_id'].unique()})
    result = all_users.merge(avg_diffs, on='user_id', how='left')
    result['user_avg_order_interval_hours'] = result['user_avg_order_interval_hours'].fillna(compute_median_time_diff(orders))
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_avg_order_interval_hours'])
    return result

def purchase_hour_cyclic(orders):
    """Apply sine-cosine transformation to the most frequent purchase hour.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_frequent_hour_sin, user_frequent_hour_cos].
    """
    hours = most_frequent_purchase_hour(orders)
    sin_vals, cos_vals = cyclic_transform(hours['user_frequent_purchase_hour'], 24)
    
    result = hours[['user_id']].copy()
    result['user_frequent_hour_sin'] = sin_vals
    result['user_frequent_hour_cos'] = cos_vals
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_frequent_hour_sin', 'user_frequent_hour_cos'])
    return result

def purchase_season_cyclic(orders):
    """Apply sine-cosine transformation to the most frequent purchase month.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, user_frequent_season_sin, user_frequent_season_cos].
    """
    orders_with_month = orders[['user_id', 'order_date']].copy()
    orders_with_month['month'] = orders_with_month['order_date'].dt.month
    
    month_counts = orders_with_month.groupby(['user_id', 'month']).size().reset_index(name='count')
    idx = month_counts.groupby('user_id')['count'].idxmax()
    most_frequent_month = month_counts.loc[idx, ['user_id', 'month']].rename(columns={'month': 'most_frequent_month'})
    
    all_users = pd.DataFrame({'user_id': orders['user_id'].unique()})
    most_frequent_month = all_users.merge(most_frequent_month, on='user_id', how='left').fillna({'most_frequent_month': 1})  # Default to January
    
    sin_vals, cos_vals = cyclic_transform(most_frequent_month['most_frequent_month'], 12)
    result = most_frequent_month[['user_id']].copy()
    result['user_frequent_season_sin'] = sin_vals
    result['user_frequent_season_cos'] = cos_vals
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['user_id', 'user_frequent_season_sin', 'user_frequent_season_cos'])
    return result

########################## Feature Engineering: Order-Level Features ######################
# Each feature returns a DataFrame with columns: order_id, feature_name.

def add_feature_order_contains_alcohol(order_products_denormalized, orders):
    """Flag orders containing alcohol products.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_has_alcohol].
    """
    alcohol_orders = order_products_denormalized[order_products_denormalized['department'] == 'alcohol'][['order_id']].drop_duplicates()
    alcohol_orders['order_has_alcohol'] = 1
    
    all_orders = pd.DataFrame({'order_id': orders['order_id'].unique()})
    result = all_orders.merge(alcohol_orders, on='order_id', how='left').fillna({'order_has_alcohol': 0})
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_has_alcohol'])
    return result

def add_feature_order_item_count(order_products_denormalized):
    """Count total items in each order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_product_count].
    """
    item_counts = order_products_denormalized.groupby('order_id')['product_id'].count().reset_index(name='order_product_count')
    item_counts = downcast_dtypes(item_counts)
    
    validate_dataframe(item_counts, ['order_id', 'order_product_count'])
    return item_counts

def add_feature_order_unique_departments_count(order_products_denormalized):
    """Count unique departments in each order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_unique_dept_count].
    """
    dept_counts = order_products_denormalized.groupby('order_id')['department'].nunique().reset_index(name='order_unique_dept_count')
    dept_counts = downcast_dtypes(dept_counts)
    
    validate_dataframe(dept_counts, ['order_id', 'order_unique_dept_count'])
    return dept_counts

def add_feature_order_unique_aisles_count(order_products_denormalized):
    """Count unique aisles in each order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_unique_aisle_count].
    """
    aisle_counts = order_products_denormalized.groupby('order_id')['aisle'].nunique().reset_index(name='order_unique_aisle_count')
    aisle_counts = downcast_dtypes(aisle_counts)
    
    validate_dataframe(aisle_counts, ['order_id', 'order_unique_aisle_count'])
    return aisle_counts

def add_feature_order_unique_departments_ratio(order_products_denormalized):
    """Calculate ratio of unique departments to total items in each order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, 'order_unique_dept_ratio'].
    """
    total_items = add_feature_order_item_count(order_products_denormalized)
    unique_depts = add_feature_order_unique_departments_count(order_products_denormalized)
    merged = total_items.merge(unique_depts, on='order_id')
    merged['order_unique_dept_ratio'] = merged['order_unique_dept_count'] / merged['order_product_count']
    merged = downcast_dtypes(merged)
    
    validate_dataframe(merged, ['order_id', 'order_unique_dept_ratio'])
    return merged[['order_id', 'order_unique_dept_ratio']]

def add_feature_order_unique_aisles_ratio(order_products_denormalized):
    """Calculate ratio of unique aisles to total items in each order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_unique_aisle_ratio].
    """
    total_items = add_feature_order_item_count(order_products_denormalized)
    unique_aisles = add_feature_order_unique_aisles_count(order_products_denormalized)
    merged = total_items.merge(unique_aisles, on='order_id')
    merged['order_unique_aisle_ratio'] = merged['order_unique_aisle_count'] / merged['order_product_count']
    merged = downcast_dtypes(merged)
    
    validate_dataframe(merged, ['order_id', 'order_unique_aisle_ratio'])
    return merged[['order_id', 'order_unique_aisle_ratio']]

def add_feature_avg_tip_rate_department(order_products_denormalized, orders, tips_public, default_tip_rate=0.500111):
    """Calculate the average tip rate for each department up to the order date, aggregated per order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
        tips_public (pd.DataFrame): Tips dataset.
        default_tip_rate (float): Default tip rate for departments with no prior orders.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_dept_tip_rate].
    """
    merged = order_products_denormalized.merge(
        orders[['order_id', 'order_date']], on='order_id'
    ).merge(
        tips_public[['order_id', 'tip']], on='order_id', how='left'
    )
    
    merged['tip'] = merged['tip'].fillna(0).astype('float32')
    merged = merged.sort_values(by=['department', 'order_date'])
    
    merged['order_count'] = merged.groupby('department').cumcount().astype('int32')
    merged['tip_cumsum_before'] = merged.groupby('department')['tip'].cumsum() - merged['tip']
    merged['avg_tip_rate_before'] = merged['tip_cumsum_before'] / merged['order_count']
    merged.loc[merged['order_count'] == 0, 'avg_tip_rate_before'] = pd.NA
    
    order_tip_rate = merged.groupby('order_id')['avg_tip_rate_before'].mean().reset_index(name='order_dept_tip_rate')
    order_tip_rate['order_dept_tip_rate'] = order_tip_rate['order_dept_tip_rate'].fillna(default_tip_rate).astype('float32')
    order_tip_rate = downcast_dtypes(order_tip_rate)
    
    validate_dataframe(order_tip_rate, ['order_id', 'order_dept_tip_rate'])
    return order_tip_rate

def add_feature_avg_tip_rate_aisle(order_products_denormalized, orders, tips_public, default_tip_rate=0.500111):
    """Calculate the average tip rate for each aisle up to the order date, aggregated per order.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
        tips_public (pd.DataFrame): Tips dataset.
        default_tip_rate (float): Default tip rate for aisles with no prior orders.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_aisle_tip_rate].
    """
    merged = order_products_denormalized.merge(
        orders[['order_id', 'order_date']], on='order_id'
    ).merge(
        tips_public[['order_id', 'tip']], on='order_id', how='left'
    )
    
    merged['tip'] = merged['tip'].fillna(0).astype('float32')
    merged = merged.sort_values(by=['aisle', 'order_date'])
    
    merged['order_count'] = merged.groupby('aisle').cumcount().astype('int32')
    merged['tip_cumsum_before'] = merged.groupby('aisle')['tip'].cumsum() - merged['tip']
    merged['avg_tip_rate_before'] = merged['tip_cumsum_before'] / merged['order_count']
    merged.loc[merged['order_count'] == 0, 'avg_tip_rate_before'] = pd.NA
    
    order_tip_rate = merged.groupby('order_id')['avg_tip_rate_before'].mean().reset_index(name='order_aisle_tip_rate')
    order_tip_rate['order_aisle_tip_rate'] = order_tip_rate['order_aisle_tip_rate'].fillna(default_tip_rate).astype('float32')
    order_tip_rate = downcast_dtypes(order_tip_rate)
    
    validate_dataframe(order_tip_rate, ['order_id', 'order_aisle_tip_rate'])
    return order_tip_rate

def order_hour(orders):
    """Extract the hour of the day for each order.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_placed_hour].
    """
    result = orders[['order_id', 'order_date']].copy()
    result['order_placed_hour'] = result['order_date'].dt.hour
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_placed_hour'])
    return result[['order_id', 'order_placed_hour']]

def order_dow(orders):
    """Extract the day of week for each order (0=Monday, 6=Sunday).
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_placed_dow].
    """
    result = orders[['order_id', 'order_date']].copy()
    result['order_placed_dow'] = result['order_date'].dt.dayofweek
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_placed_dow'])
    return result[['order_id', 'order_placed_dow']]

def is_weekend_order(orders):
    """Flag orders placed on weekends (Saturday or Sunday).
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_is_weekend].
    """
    dow = order_dow(orders)
    result = dow[['order_id']].copy()
    result['order_is_weekend'] = dow['order_placed_dow'].isin([5, 6]).astype('int8')
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_is_weekend'])
    return result

def order_hour_cyclic(orders):
    """Apply sine-cosine transformation to the order hour.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_placed_hour_sin, order_placed_hour_cos].
    """
    hours = order_hour(orders)
    sin_vals, cos_vals = cyclic_transform(hours['order_placed_hour'], 24)
    
    result = hours[['order_id']].copy()
    result['order_placed_hour_sin'] = sin_vals
    result['order_placed_hour_cos'] = cos_vals
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_placed_hour_sin', 'order_placed_hour_cos'])
    return result

def order_season_cyclic(orders):
    """Apply sine-cosine transformation to the order month.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_placed_season_sin, order_placed_season_cos].
    """
    result = orders[['order_id', 'order_date']].copy()
    result['month'] = result['order_date'].dt.month
    
    sin_vals, cos_vals = cyclic_transform(result['month'], 12)
    result['order_placed_season_sin'] = sin_vals
    result['order_placed_season_cos'] = cos_vals
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_placed_season_sin', 'order_placed_season_cos'])
    return result[['order_id', 'order_placed_season_sin', 'order_placed_season_cos']]

def time_since_last_order(orders):
    """Calculate time since the user's last order in hours.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [order_id, order_time_since_last_hours].
    """
    sorted_orders = orders[['order_id', 'user_id', 'order_date']].sort_values(['user_id', 'order_date'])
    time_diffs = sorted_orders.groupby('user_id')['order_date'].diff().shift(-1).dt.total_seconds() / 3600
    time_diffs = time_diffs.reindex(sorted_orders.index).shift(1)  # Shift back to align with current order
    
    result = sorted_orders[['order_id']].copy()
    result['order_time_since_last_hours'] = time_diffs.fillna(compute_median_time_diff(orders))
    result = downcast_dtypes(result)
    
    validate_dataframe(result, ['order_id', 'order_time_since_last_hours'])
    return result

########################## Feature Engineering: User-Product-Level Features ##############
# Each feature returns a DataFrame with columns: user_id, product_id, feature_name.

def count_products_per_user(order_products_denormalized, orders):
    """Count how many times each user purchased each product.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
    
    Returns:
        pd.DataFrame: Columns [user_id, product_id, user_product_purchase_count].
    """
    merged = merge_with_orders(order_products_denormalized, orders)
    counts = merged.groupby(['user_id', 'product_id']).size().reset_index(name='user_product_purchase_count')
    counts = downcast_dtypes(counts)
    
    validate_dataframe(counts, ['user_id', 'product_id', 'user_product_purchase_count'])
    return counts

def create_user_product_tip_probability(order_products_denormalized, orders, tips_public, default_tip_rate=0.500111):
    """Calculate the average tip probability for each user-product pair, aggregated to order_id.
    
    Args:
        order_products_denormalized (pd.DataFrame): Order products dataset.
        orders (pd.DataFrame): Orders dataset.
        tips_public (pd.DataFrame): Tips dataset.
        default_tip_rate (float): Default tip probability for pairs with no history.
    
    Returns:
        pd.DataFrame: Columns [order_id, user_product_tip_prob].
    """
    # Validierung der Eingabedaten
    if 'user_id' not in orders.columns:
        raise ValueError("Spalte 'user_id' fehlt in orders DataFrame")
    if 'order_id' not in order_products_denormalized.columns:
        raise ValueError("Spalte 'order_id' fehlt in order_products_denormalized DataFrame")
    
    # Merge der DataFrames, um user_id hinzuzufügen
    merged = order_products_denormalized.merge(
        orders[['order_id', 'user_id', 'order_date']], 
        on='order_id', 
        how='left'
    )
    
    # Validierung nach erstem Merge
    if 'user_id' not in merged.columns:
        raise ValueError("user_id fehlt nach Merge mit orders. Überprüfe order_id Übereinstimmungen.")
    if merged['user_id'].isna().any():
        print(f"Warnung: {merged['user_id'].isna().sum()} Zeilen mit fehlendem user_id nach Merge mit orders")
        merged = merged.dropna(subset=['user_id'])
    
    merged = merged.merge(
        tips_public[['order_id', 'tip']], 
        on='order_id', 
        how='left'
    )
    
    # Optimierung und Berechnung
    merged['tip'] = merged['tip'].fillna(0).astype('float32')
    merged = merged.sort_values(by=['user_id', 'product_id', 'order_date'])
    
    merged['times_bought_before'] = merged.groupby(['user_id', 'product_id']).cumcount().astype('int32')
    merged['tip_cumsum_before'] = merged.groupby(['user_id', 'product_id'])['tip'].cumsum() - merged['tip']
    merged['avg_tip_rate_before'] = merged['tip_cumsum_before'] / merged['times_bought_before']
    merged.loc[merged['times_bought_before'] == 0, 'avg_tip_rate_before'] = pd.NA
    
    # Aggregation zu order_id Ebene
    order_tip_prob = merged.groupby('order_id')['avg_tip_rate_before'].mean().reset_index(
        name='user_product_tip_prob'
    )
    order_tip_prob['user_product_tip_prob'] = order_tip_prob['user_product_tip_prob'].fillna(default_tip_rate).astype('float32')
    order_tip_prob = downcast_dtypes(order_tip_prob)
    
    # Validierung
    expected_columns = ['order_id', 'user_product_tip_prob']
    validate_dataframe(order_tip_prob, expected_columns)
    
    return order_tip_prob

################ Combine all Features ####################################

def combine_all_features(orders, order_products_denormalized, tips_public, default_tip_rate=0.500111):
    """Combine all engineered features into a DataFrame on order_id level.
    
    Args:
        orders (pd.DataFrame): Orders dataset.
        order_products_denormalized (pd.DataFrame): Order products dataset.
        tips_public (pd.DataFrame): Tips dataset.
        default_tip_rate (float): Default tip probability for missing values.
    
    Returns:
        pd.DataFrame: DataFrame with all features, keyed by order_id, and including the tip column.
    """
    # Base DataFrame with orders (order_id, user_id)
    base_df = orders[['order_id', 'user_id']].copy()
    base_df['order_id'] = base_df['order_id'].astype('int64')
    base_df['user_id'] = base_df['user_id'].astype('int64')
    base_df = downcast_dtypes(base_df)
    
    # Validierung
    expected_rows = len(orders)
    if len(base_df) != expected_rows:
        raise ValueError(f"Base DataFrame hat {len(base_df)} Zeilen, erwartet {expected_rows}")
    
    # Merge order-level features
    order_features = [
        add_feature_order_contains_alcohol(order_products_denormalized, orders),
        add_feature_order_item_count(order_products_denormalized),
        add_feature_order_unique_departments_count(order_products_denormalized),
        add_feature_order_unique_aisles_count(order_products_denormalized),
        add_feature_order_unique_departments_ratio(order_products_denormalized),
        add_feature_order_unique_aisles_ratio(order_products_denormalized),
        add_feature_avg_tip_rate_department(order_products_denormalized, orders, tips_public, default_tip_rate),
        add_feature_avg_tip_rate_aisle(order_products_denormalized, orders, tips_public, default_tip_rate),
        order_hour(orders),
        order_dow(orders),
        is_weekend_order(orders),
        order_hour_cyclic(orders),
        order_season_cyclic(orders),
        time_since_last_order(orders)
    ]
    
    result = base_df
    for feature_df in order_features:
        if 'order_id' in feature_df.columns:
            feature_df['order_id'] = feature_df['order_id'].astype('int64')
            # print(f"Merging feature with columns {feature_df.columns.tolist()}, order_id dtype: {feature_df['order_id'].dtype}")
        else:
            raise ValueError(f"feature_df missing order_id: {feature_df.columns.tolist()}")
        
        merge_cols = ['order_id']
        feature_cols = [col for col in feature_df.columns if col not in merge_cols]
        result_cols = [col for col in result.columns if col in feature_cols]
        if result_cols:
            result = result.drop(columns=result_cols)
        result = result.merge(feature_df, on='order_id', how='left')
        result = downcast_dtypes(result)
        
        # Validierung nach Merge
        if len(result) != len(base_df):
            raise ValueError(f"Zeilenanzahl nach Merge mit {feature_df.columns.tolist()} geändert: {len(result)}")
    
    # Merge user-level features
    user_features = [
        add_feature_alcohol_count(order_products_denormalized, orders),
        total_products_per_user(order_products_denormalized, orders),
        total_unique_products_per_user(order_products_denormalized, orders),
        unique_to_total_product_ratio_per_user(order_products_denormalized, orders),
        most_frequent_purchase_hour(orders),
        most_frequent_purchase_dow(orders),
        avg_time_between_orders(orders),
        purchase_hour_cyclic(orders),
        purchase_season_cyclic(orders)
    ]
    
    for feature_df in user_features:
        if 'user_id' in feature_df.columns:
            feature_df['user_id'] = feature_df['user_id'].astype('int64')
            # print(f"Merging user feature with columns {feature_df.columns.tolist()}, user_id dtype: {feature_df['user_id'].dtype}")
        else:
            raise ValueError(f"feature_df missing user_id: {feature_df.columns.tolist()}")
        
        merge_cols = ['user_id']
        feature_cols = [col for col in feature_df.columns if col not in merge_cols]
        result_cols = [col for col in result.columns if col in feature_cols]
        if result_cols:
            result = result.drop(columns=result_cols)
        result = result.merge(feature_df, on='user_id', how='left')
        result = downcast_dtypes(result)
        
        # Validierung nach Merge
        if len(result) != len(base_df):
            raise ValueError(f"Zeilenanzahl nach Merge mit {feature_df.columns.tolist()} geändert: {len(result)}")
    
    # Merge user-product-level features (aggregated to order_id)
    user_product = count_products_per_user(order_products_denormalized, orders)
    user_product_sum = user_product.groupby('user_id')['user_product_purchase_count'].sum().reset_index(name='user_total_product_purchase_count')
    user_product_sum['user_id'] = user_product_sum['user_id'].astype('int64')
    result = result.merge(user_product_sum[['user_id', 'user_total_product_purchase_count']], on='user_id', how='left')
    result['user_total_product_purchase_count'] = result['user_total_product_purchase_count'].fillna(0).astype('int32')
    result = downcast_dtypes(result)
    
    # Add user-product tip probability (aggregated to order_id)
    user_product_tip_df = create_user_product_tip_probability(order_products_denormalized, orders, tips_public, default_tip_rate)
    user_product_tip_df['order_id'] = user_product_tip_df['order_id'].astype('int64')
    result = result.merge(
        user_product_tip_df[['order_id', 'user_product_tip_prob']],
        on='order_id',
        how='left'
    )
    result['user_product_tip_prob'] = result['user_product_tip_prob'].fillna(default_tip_rate).astype('float32')
    result = downcast_dtypes(result)
    
    # Add target variable (tip)
    result = result.merge(tips_public[['order_id', 'tip']], on='order_id', how='left')
    result = downcast_dtypes(result)
    
    # Validierung
    if len(result) != expected_rows:
        raise ValueError(f"Endgültige Zeilenanzahl stimmt nicht: {len(result)} statt {expected_rows}")
    
    expected_columns = [
        'order_id', 'user_id', 'order_has_alcohol', 'order_product_count', 
        'order_unique_dept_count', 'order_unique_aisle_count', 'order_unique_dept_ratio', 
        'order_unique_aisle_ratio', 'order_dept_tip_rate', 'order_aisle_tip_rate', 
        'order_placed_hour', 'order_placed_dow', 'order_is_weekend', 'order_placed_hour_sin', 
        'order_placed_hour_cos', 'order_placed_season_sin', 'order_placed_season_cos', 
        'order_time_since_last_hours', 'user_alcohol_purchase_count', 'user_total_purchase_count', 
        'user_unique_product_count', 'user_unique_to_total_ratio', 'user_frequent_purchase_hour', 
        'user_frequent_purchase_dow', 'user_avg_order_interval_hours', 'user_frequent_hour_sin', 
        'user_frequent_hour_cos', 'user_frequent_season_sin', 'user_frequent_season_cos', 
        'user_total_product_purchase_count', 'user_product_tip_prob', 'tip'
    ]
    validate_dataframe(result, expected_columns)
    
    return result