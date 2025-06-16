from prefect import flow, task
import sqlite3
import pandas as pd

# Feature Engineering and Forecasting from your modules
from features import combine_all_features
from Tip_forecasting import process_tip_time_series, plot_forecast_with_split

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DB_NAME = "dabi2_projekt.db"

# --------------------------------------------
# 1. Create SQLite database structure
# --------------------------------------------
@task
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create tables only if they do not exist (safe to rerun)
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS user (
            user_id INTEGER PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            date DATE,
            tip INTEGER,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(user_id)
        );

        CREATE TABLE IF NOT EXISTS department (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT
        );

        CREATE TABLE IF NOT EXISTS aisle (
            aisle_id INTEGER PRIMARY KEY,
            aisle_name TEXT,
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES department(department_id)
        );

        CREATE TABLE IF NOT EXISTS product (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            aisle_id INTEGER,
            FOREIGN KEY (aisle_id) REFERENCES aisle(aisle_id)
        );

        CREATE TABLE IF NOT EXISTS invoice (
            order_id INTEGER,
            product_id INTEGER,
            add_to_cart_order INTEGER,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES product(product_id)
        );
    """)
    conn.commit()
    conn.close()

# --------------------------------------------
# 2. Load data from CSV/Parquet files
# --------------------------------------------
@task
def load_data():
    order_product = pd.read_csv('order_products_denormalized.csv')
    orders = pd.read_parquet('orders.parquet')
    tips = pd.read_csv('tips_public.csv').drop(columns=["Unnamed: 0"], errors='ignore')
    return order_product, orders, tips

# --------------------------------------------
# 3. Insert data into the SQLite database (only if empty)
# --------------------------------------------
@task
def populate_tables(order_product, orders, tips):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    def table_is_empty(table_name):
        result = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] == 0

    m_tip_order = pd.merge(orders, tips, on='order_id', how='left')

    aisles = order_product[['aisle_id', 'aisle', 'department_id']].drop_duplicates().rename(columns={'aisle': 'aisle_name'})
    departments = order_product[['department_id', 'department']].drop_duplicates().rename(columns={'department': 'department_name'})
    products = order_product[['product_id', 'product_name', 'aisle_id']].drop_duplicates()
    users = orders[['user_id']].drop_duplicates()
    orders_clean = m_tip_order[['order_id', 'order_date', 'user_id', 'tip']].rename(columns={'order_date': 'date'})
    invoices = order_product[['order_id', 'product_id', 'add_to_cart_order']]

    if table_is_empty('department'):
        departments.to_sql('department', conn, if_exists='append', index=False)
    if table_is_empty('aisle'):
        aisles.to_sql('aisle', conn, if_exists='append', index=False)
    if table_is_empty('product'):
        products.to_sql('product', conn, if_exists='append', index=False)
    if table_is_empty('user'):
        users.to_sql('user', conn, if_exists='append', index=False)
    if table_is_empty('orders'):
        orders_clean.to_sql('orders', conn, if_exists='append', index=False)
    if table_is_empty('invoice'):
        invoices.to_sql('invoice', conn, if_exists='append', index=False)

    conn.close()

# --------------------------------------------
# 4. Run full feature engineering
# --------------------------------------------
@task
def feature_engineering(order_product, orders, tips):
    df = combine_all_features(orders, order_product, tips)
    df.to_csv("all_features.csv", index=False)
    return df

# --------------------------------------------
# 5. Machine learning to predict "tip" (yes/no)
# --------------------------------------------
@task
def predict_tip(df):
    train_df = df[~df['tip'].isna()].copy()
    predict_df = df[df['tip'].isna()].copy()

    print(f"\n🔢 Training data: {train_df.shape}")
    print(f"🔍 Prediction data: {predict_df.shape}")

    features = [col for col in df.columns if col not in ['order_id', 'user_id', 'tip']]
    X = train_df[features]
    y = train_df['tip'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Classification report:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if not predict_df.empty:
        X_pred = predict_df[features]
        predict_df['tip'] = model.predict(X_pred).astype(int)
        predict_df[['order_id', 'tip']].to_csv("predicted_tips.csv", index=False)
        print("\n✅ Predictions saved to 'predicted_tips.csv'")

# --------------------------------------------
# 6. Time series forecast for tip share
# --------------------------------------------
@task
def run_forecasting(orders, tips):
    forecast_df = process_tip_time_series(
        orders, tips, min_orders=30, forecast_hours=24 * 7
    )
    plot_forecast_with_split(forecast_df)

# --------------------------------------------
# 7. Main Prefect flow – combines everything
# --------------------------------------------
@flow(name="DABI2 Full Process with Inline Model")
def full_pipeline():
    create_tables()
    order_product, orders, tips = load_data()
    populate_tables(order_product, orders, tips)
    all_features = feature_engineering(order_product, orders, tips)
    predict_tip(all_features)
    run_forecasting(orders, tips)

# --------------------------------------------
# 8. Start the flow locally or via CLI
# --------------------------------------------
if __name__ == "__main__":
    full_pipeline.serve(name="dabi2-pipeline-inline-model")
