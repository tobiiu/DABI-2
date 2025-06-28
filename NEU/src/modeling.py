import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Define the list of features your model will use.
# This list is now centralized here for consistency.
FEATURE_COLUMNS = [
    'order_has_alcohol', 'order_product_count', 'order_unique_dept_count',
    'order_unique_aisle_count', 'order_unique_dept_ratio', 'order_unique_aisle_ratio',
    'order_dept_tip_rate', 'order_aisle_tip_rate', 'order_placed_hour',
    'order_placed_dow', 'order_is_weekend', 'order_placed_hour_sin',
    'order_placed_hour_cos', 'order_placed_season_sin', 'order_placed_season_cos',
    'order_time_since_last_hours', 'user_alcohol_purchase_count',
    'user_total_purchase_count', 'user_unique_product_count',
    'user_unique_to_total_ratio', 'user_frequent_purchase_hour',
    'user_frequent_purchase_dow', 'user_avg_order_interval_hours',
    'user_frequent_hour_sin', 'user_frequent_hour_cos',
    'user_frequent_season_sin', 'user_frequent_season_cos',
    'user_total_product_purchase_count', 'user_product_tip_prob',
    # --- NEW FEATURES WILL BE ADDED HERE ---
    'forecasted_hourly_tip_share', 'tip_lag_1', 'tip_lag_2', 'tip_lag_3'
]

def get_feature_and_target(df: pd.DataFrame, target_col: str):
    """Splits dataframe into features (X) and target (y) for training."""
    # Ensure all feature columns exist, fill missing with 0 if necessary
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            
    X = df[FEATURE_COLUMNS]
    y = df[target_col].astype('int')
    return X, y

def train_and_evaluate_model(features_df: pd.DataFrame, force_retrain: bool = False) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier and evaluates its performance.
    Loads an existing model if available, unless force_retrain is True.

    Args:
        features_df (pd.DataFrame): The dataframe with features and the 'tip' target.
        force_retrain (bool): If True, retrains the model even if a saved one exists.

    Returns:
        RandomForestClassifier: The trained or loaded model object.
    """
    model_path = 'output/tip_prediction_model.joblib'
    
    # --- MODEL PERSISTENCE LOGIC ---
    if not force_retrain and os.path.exists(model_path):
        print("Loading existing model from output/tip_prediction_model.joblib")
        model = joblib.load(model_path)
        return model

    print("--- Training a new model ---")
    train_df = features_df[~features_df['tip'].isna()].copy()
    X, y = get_feature_and_target(train_df, 'tip')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training model on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, model_path)
    print(f"\nTrained model saved to {model_path}")
    
    return model

def make_predictions(model: RandomForestClassifier, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes predictions on the portion of the data where 'tip' is NaN.

    Args:
        model (RandomForestClassifier): The trained model.
        features_df (pd.DataFrame): The dataframe containing all features.

    Returns:
        pd.DataFrame: A dataframe with 'order_id' and 'tip' predictions.
    """
    predict_df = features_df[features_df['tip'].isna()].copy()
    
    if predict_df.empty:
        print("No missing tips to predict.")
        return pd.DataFrame(columns=['order_id', 'tip'])

    # *** FIX: Directly select feature columns without trying to process a non-existent y ***
    X_predict = predict_df[FEATURE_COLUMNS]
    # Fill any potential NaNs in the features before predicting
    X_predict = X_predict.fillna(0)
    
    print(f"\nMaking predictions for {len(X_predict)} orders...")
    predictions = model.predict(X_predict)
    
    result_df = pd.DataFrame({
        'order_id': predict_df['order_id'],
        'tip': predictions
    })
    
    result_df.to_csv('output/predicted_tips.csv', index=False)
    print("Predictions saved to output/predicted_tips.csv")
    
    return result_df