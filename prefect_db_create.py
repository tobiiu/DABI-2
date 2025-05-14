from prefect import flow, task
import sqlite3
import pandas as pd
import os

@task
def create_tables():
    conn = sqlite3.connect('dabi2_projekt.db')
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS user')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            user_id INTEGER PRIMARY KEY
        )
    ''')

    cursor.execute('DROP TABLE IF EXISTS orders')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            date DATE,
            tip INTEGER,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(user_id)
        )
    ''')

    cursor.execute('DROP TABLE IF EXISTS department')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS department (
            department_id INTEGER PRIMARY KEY,
            department_name VARCHAR(160)
        )
    ''')

    cursor.execute('DROP TABLE IF EXISTS aisle')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aisle (
            aisle_id INTEGER PRIMARY KEY,
            aisle_name VARCHAR(160),
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES department(department_id)
        )
    ''')

    cursor.execute('DROP TABLE IF EXISTS product')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS product (
            product_id INTEGER PRIMARY KEY,
            product_name VARCHAR(160),
            aisle_id INTEGER,
            FOREIGN KEY (aisle_id) REFERENCES aisle(aisle_id)
        )
    ''')

    cursor.execute('DROP TABLE IF EXISTS invoice')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoice (
            order_id INTEGER,
            product_id INTEGER,
            add_to_cart_order INTEGER,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES product(product_id)
        )
    ''')

    conn.commit()
    conn.close()

@task
def load_data():
    order_product = pd.read_csv('order_products_denormalized.csv')
    orders = pd.read_parquet('orders.parquet')
    tips = pd.read_csv('tips_public.csv')
    return order_product, orders, tips

@task
def populate_tables(order_product, orders, tips):
    conn = sqlite3.connect('dabi2_projekt.db')

    # Mergen
    m_tip_order = pd.merge(orders, tips, on='order_id', how='left')

    aisles = (
        order_product[['aisle_id', 'aisle', 'department_id']]
        .drop_duplicates(subset=['aisle_id'])
        .rename(columns={'aisle': 'aisle_name'})
    )

    department = (
        order_product[['department_id', 'department']]
        .drop_duplicates(subset=['department_id'])
        .rename(columns={'department': 'department_name'})
    )

    product = order_product[['product_id', 'product_name', 'aisle_id']].drop_duplicates()
    user = orders[['user_id']].drop_duplicates()
    order = (
        m_tip_order[['order_id', 'order_date', 'user_id', 'tip']]
        .drop_duplicates()
        .rename(columns={'order_date': 'date'})
    )
    invoice = order_product[['order_id', 'product_id', 'add_to_cart_order']]

    department.to_sql('department', conn, if_exists='append', index=False)
    aisles.to_sql('aisle', conn, if_exists='append', index=False)
    product.to_sql('product', conn, if_exists='append', index=False)
    user.to_sql('user', conn, if_exists='append', index=False)
    order.to_sql('orders', conn, if_exists='append', index=False)
    invoice.to_sql('invoice', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()

@flow
def main_flow():
    create_tables()  
    order_product, orders, tips = load_data()
    populate_tables(order_product, orders, tips)  
if __name__ == "__main__":
    main_flow.serve(name="my-local-deployment_2")
