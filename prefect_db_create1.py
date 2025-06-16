from prefect import flow, task
import sqlite3
import pandas as pd
import os

# Optional: Feature & ML Import
from features import combine_all_features
from model_tip_prediction import train_and_predict_model
from Tip_forecasting import process_tip_time_series, plot_forecast_with_split


DB_NAME = "dabi2_projekt.db"


@task
def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Drop + Create Statements
    cursor.executescript('''
        DROP TABLE IF EXISTS invoice;
        DROP TABLE IF EXISTS product;
        DROP TABLE IF EXISTS aisle;
        DROP TABLE IF EXISTS department;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS user;

        CREATE TABLE user (user_id INTEGER PRIMARY KEY);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            date DATE,
            tip INTEGER,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(user_id)
        );
        CREATE TABLE department (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT
        );
        CREATE TABLE aisle (
            aisle_id INTEGER PRIMARY KEY,
            aisle_name TEXT,
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES department(department_id)
        );
        CREATE TABLE product (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            aisle_id INTEGER,
            FOREIGN KEY (aisle_id) REFERENCES aisle(aisle_id)
        );
        CREATE TABLE invoice (
            order_id INTEGER,
            product_id INTEGER,
            add_to_cart_order INTEGER,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES product(product_id)
        );
    ''')

    conn.commit()
    conn.close()


@task
def load_data():
    order_product = pd.read_csv('order_products_denormalized.csv')
    orders = pd.read_parquet('orders.parquet')
    tips = pd.read_csv('tips_public.csv').drop(columns=["Unnamed: 0"], errors="ignore")
    return order_product, orders, tips


@task
def populate_tables(order_product, orders, tips):
    conn = sqlite3.connect(DB_NAME)

    # Merge orders + tips
    m_tip_order = pd.merge(orders, tips, on='order_id', how='left')

    # Preparation
    aisles = (
        order_product[['aisle_id', 'aisle', 'department_id']]
        .drop_duplicates()
        .rename(columns={'aisle': 'aisle_name'})
    )
    departments = (
        order_product[['department_id', 'department']]
        .drop_duplicates()
        .rename(columns={'department': 'department_name'})
    )
    products = order_product[['product_id', 'product_name', 'aisle_id']].drop_duplicates()
    users = orders[['user_id']].drop_duplicates()
    orders_clean = m_tip_order[['order_id', 'order_date', 'user_id', 'tip']] \
        .rename(columns={'order_date': 'date'})
    invoices = order_product[['order_id', 'product_id', 'add_to_cart_order']]

    # Load to DB
    departments.to_sql('department', conn, if_exists='append', index=False)
    aisles.to_sql('aisle', conn, if_exists='append', index=False)
    products.to_sql('product', conn, if_exists='append', index=False)
    users.to_sql('user', conn, if_exists='append', index=False)
    orders_clean.to_sql('orders', conn, if_exists='append', index=False)
    invoices.to_sql('invoice', conn, if_exists='append', index=False)

    conn.close()


@task
def feature_engineering(order_product, orders, tips):
    df = combine_all_features(orders, order_product, tips)
    df.to_csv('all_features.csv', index=False)
    return df


@task
def model_prediction(all_features_df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    train_df = all_features_df[~all_features_df['tip'].isna()]
    predict_df = all_features_df[all_features_df['tip'].isna()]
    X = train_df.drop(columns=['tip', 'order_id', 'user_id'])
    y = train_df['tip'].astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if not predict_df.empty:
        pred_X = predict_df.drop(columns=['tip', 'order_id', 'user_id'])
        predict_df['tip'] = model.predict(pred_X)
        predict_df[['order_id', 'tip']].to_csv("predicted_tips.csv", index=False)


@task
def run_forecasting(orders, tips):
    result_df = process_tip_time_series(
        orders_df=orders,
        tips_df=tips,
        min_orders=30,
        forecast_hours=24*7
    )
    plot_forecast_with_split(result_df)


@flow(name="DABI2 Gesamtprozess Flow")
def full_pipeline():
    create_tables()
    order_product, orders, tips = load_data()
    populate_tables(order_product, orders, tips)
    features_df = feature_engineering(order_product, orders, tips)
    model_prediction(features_df)
    run_forecasting(orders, tips)


if __name__ == "__main__":
    full_pipeline.serve(name="my-local-dabi2-pipeline")
