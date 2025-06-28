import pandas as pd
import sqlite3
import os

def create_dwh_tables(db_path='data/dwh.db'):
    """
    Erstellt die Tabellen für ein einfaches Data Warehouse (Sternschema) in SQLite.
    
    Args:
        db_path (str): Pfad zur SQLite-Datenbankdatei.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True) # Stelle sicher, dass das Verzeichnis existiert
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Dimensionstabellen erstellen
    # Dim_Time: Zeitdimension
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dim_time (
            time_id INTEGER PRIMARY KEY,
            order_date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week TEXT NOT NULL,
            month INTEGER NOT NULL,
            year INTEGER NOT NULL,
            is_weekend INTEGER NOT NULL
        );
    ''')

    # Dim_User: Benutzerdimension
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dim_user (
            user_id INTEGER PRIMARY KEY
            -- Weitere user-spezifische Attribute könnten hier hinzugefügt werden,
            -- wenn sie in den Quelldaten verfügbar wären (z.B. user_demographics)
        );
    ''')

    # Dim_Product: Produktdimension
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dim_product (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            aisle_id INTEGER,
            aisle_name TEXT,
            department_id INTEGER,
            department_name TEXT
        );
    ''')

    # Fact_Orders: Faktentabelle
    # Diese Tabelle speichert die Kennzahlen (Metrics) und verknüpft sie mit den Dimensionen
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            time_id INTEGER,
            tip_given INTEGER, -- 1 if tipped, 0 if not
            order_product_count INTEGER,
            order_total_products INTEGER, -- Gesamtzahl der Produkte in dieser Bestellung
            -- Weitere Kennzahlen wie Bestellwert, Lieferkosten etc. könnten hier hinzugefügt werden
            FOREIGN KEY (user_id) REFERENCES dim_user(user_id),
            FOREIGN KEY (time_id) REFERENCES dim_time(time_id)
        );
    ''')
    
    # Fact_Order_Products: Eine weitere Faktentabelle für individuelle Artikel in Bestellungen
    # Dies ist nützlich, wenn man Kennzahlen pro Produkt innerhalb einer Bestellung analysieren möchte
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_order_products (
            order_product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            product_id INTEGER,
            add_to_cart_order INTEGER,
            FOREIGN KEY (order_id) REFERENCES fact_orders(order_id),
            FOREIGN KEY (product_id) REFERENCES dim_product(product_id)
        );
    ''')

    conn.commit()
    conn.close()
    print(f"Data Warehouse-Tabellen in '{db_path}' erfolgreich erstellt oder aktualisiert.")


def load_data_to_dwh(cleaned_data: dict, db_path='data/dwh.db'):
    """
    Lädt die bereinigten Daten in das Data Warehouse.
    
    Args:
        cleaned_data (dict): Dictionary mit den bereinigten DataFrames ('orders', 'tips_public', 'order_products').
        db_path (str): Pfad zur SQLite-Datenbankdatei.
    """
    conn = sqlite3.connect(db_path)
    
    orders_df = cleaned_data['orders'].copy()
    tips_public_df = cleaned_data['tips_public'].copy()
    order_products_df = cleaned_data['order_products'].copy()

    # --- Dim_Time befüllen ---
    print("Befülle dim_time...")
    # Erstelle eine einzigartige Liste von Datums-/Zeit-Kombinationen aus orders
    time_data = orders_df[['order_date']].drop_duplicates().copy()
    time_data['hour'] = time_data['order_date'].dt.hour
    time_data['day_of_week'] = time_data['order_date'].dt.day_name()
    time_data['month'] = time_data['order_date'].dt.month
    time_data['year'] = time_data['order_date'].dt.year
    time_data['is_weekend'] = time_data['order_date'].dt.weekday.isin([5, 6]).astype(int)
    
    # Erstelle eine time_id
    time_data['time_id'] = (time_data['order_date'].astype(int) / 10**9).astype(int) # Einfache Hash-basierte ID
    
    time_data_to_load = time_data[['time_id', 'order_date', 'hour', 'day_of_week', 'month', 'year', 'is_weekend']]
    time_data_to_load.to_sql('dim_time', conn, if_exists='append', index=False)
    print("dim_time befüllt.")

    # --- Dim_User befüllen ---
    print("Befülle dim_user...")
    user_data = orders_df[['user_id']].drop_duplicates()
    user_data.to_sql('dim_user', conn, if_exists='append', index=False)
    print("dim_user befüllt.")

    # --- Dim_Product befüllen ---
    print("Befülle dim_product...")
    product_data = order_products_df[['product_id', 'product_name', 'aisle_id', 'aisle_name', 'department_id', 'department_name']].drop_duplicates()
    product_data.to_sql('dim_product', conn, if_exists='append', index=False)
    print("dim_product befüllt.")

    # --- Fact_Orders befüllen ---
    print("Befülle fact_orders...")
    # Merge tips and time_id into orders_df
    fact_orders_df = orders_df.merge(tips_public_df[['order_id', 'tip']], on='order_id', how='left')
    fact_orders_df['tip_given'] = fact_orders_df['tip'].map({'yes': 1, 'no': 0}).fillna(-1).astype(int) # -1 für unbekannte Tipps
    
    # Füge die time_id hinzu
    fact_orders_df = fact_orders_df.merge(time_data[['order_date', 'time_id']], on='order_date', how='left')

    # Berechne order_product_count
    order_product_counts = order_products_df.groupby('order_id').size().reset_index(name='order_total_products')
    fact_orders_df = fact_orders_df.merge(order_product_counts, on='order_id', how='left')

    fact_orders_to_load = fact_orders_df[['order_id', 'user_id', 'time_id', 'tip_given', 'order_total_products']]
    fact_orders_to_load.to_sql('fact_orders', conn, if_exists='append', index=False)
    print("fact_orders befüllt.")

    # --- Fact_Order_Products befüllen ---
    print("Befülle fact_order_products...")
    # Sicherstellen, dass order_product_id einzigartig ist (AUTOINCREMENT in DB)
    fact_order_products_to_load = order_products_df[['order_id', 'product_id', 'add_to_cart_order']]
    fact_order_products_to_load.to_sql('fact_order_products', conn, if_exists='append', index=False)
    print("fact_order_products befüllt.")

    conn.close()
    print(f"Daten erfolgreich in das Data Warehouse '{db_path}' geladen.")


# Beispiel für eine einfache OLAP-ähnliche Abfrage
def query_dwh_example(db_path='data/dwh.db'):
    """
    Führt eine beispielhafte Abfrage auf dem Data Warehouse aus.
    Z.B. durchschnittliche Trinkgeldrate pro Wochentag.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        dt.day_of_week,
        AVG(CASE WHEN fo.tip_given = 1 THEN 1.0 ELSE 0.0 END) * 100 AS tip_percentage
    FROM
        fact_orders fo
    JOIN
        dim_time dt ON fo.time_id = dt.time_id
    WHERE
        fo.tip_given != -1 -- Nur bekannte Trinkgeldinformationen berücksichtigen
    GROUP BY
        dt.day_of_week
    ORDER BY
        tip_percentage DESC;
    """
    result = pd.read_sql_query(query, conn)
    conn.close()
    print("\nBeispielhafte DWH-Abfrage (Durchschnittliche Trinkgeldrate pro Wochentag):")
    print(result)
    return result

