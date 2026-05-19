"""
scripts/load_to_postgres.py

Establishes connection to the local PostgreSQL database, initializes schemas,
and populates the relational entities with processed review observations.
"""

import os
import sys
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Database Connection Parameters Configuration
DB_HOST = "localhost"
DB_NAME = "bank_reviews"
DB_USER = "postgres"       # Change this to your PostgreSQL user if different
DB_PASS = "postgresql"       # Change this to your PostgreSQL password if different
DB_PORT = "5432"

def load_data_to_postgres():
    csv_path = "data/processed/analyzed_bank_reviews.csv"
    
    # 1. Defensive Input File Verification Check
    if not os.path.exists(csv_path):
        print(f"❌ Error: Processed data asset '{csv_path}' missing. Run analysis script first!")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ Warning: Dataframe is empty. Aborting insertion routines.")
        return

    print("🔌 Establishing connection to PostgreSQL instance...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
        )
        cursor = conn.cursor()
    except psycopg2.OperationalError as e:
        print(f"❌ Database Connection Failure: Ensure PostgreSQL server is running. Details:\n{e}")
        sys.exit(1)

    # 2. Schema DDL Executions
    print("🏗️ Initializing database tables...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS banks (
        bank_id INT PRIMARY KEY,
        bank_name VARCHAR(100) NOT NULL UNIQUE,
        app_name VARCHAR(100) NOT NULL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id VARCHAR(50) PRIMARY KEY,
        bank_id INT NOT NULL,
        review_text TEXT NOT NULL,
        rating INT NOT NULL,
        review_date DATE NOT NULL,
        sentiment_label VARCHAR(20) NOT NULL,
        sentiment_score NUMERIC(5, 4) NOT NULL,
        identified_theme VARCHAR(100) NOT NULL,
        source VARCHAR(50) NOT NULL,
        FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
    );
    """)

    # 3. Seed Metadata Dimension Table
    print("🌱 Seeding corporate static lookup dimension records...")
    bank_metadata = [
        (1, "Commercial Bank of Ethiopia", "prod.cbe.birr"),
        (2, "Bank of Abyssinia", "com.boa.boaMobileBanking"),
        (3, "Dashen Bank", "com.dashen.dashensuperapp")
    ]
    cursor.executemany("""
        INSERT INTO banks (bank_id, bank_name, app_name) 
        VALUES (%s, %s, %s) 
        ON CONFLICT (bank_id) DO NOTHING;
    """, bank_metadata)

    # 4. Map Bank Strings to Relational Key Indexes safely
    bank_mapping = {
        "Commercial Bank of Ethiopia": 1,
        "Bank of Abyssinia": 2,
        "Dashen Bank": 3
    }
    df['bank_id'] = df['bank'].map(bank_mapping)

    # 5. Extract and Transform DataFrame tuples for explicit bulk insertion
    print(f"📥 Preparing insertion tuples for {len(df)} records...")
    review_tuples = list(df[[
        'review_id', 'bank_id', 'review', 'rating', 'date', 
        'sentiment_label', 'sentiment_score', 'identified_theme', 'source'
    ]].itertuples(index=False, name=None))

    # Clean up review records to handle truncation checks or string anomalies safely
    review_tuples = [
        (r[0], r[1], str(r[2]), int(r[3]), r[4], r[5], float(r[6]), r[7], r[8])
        for r in review_tuples if pd.notna(r[1])
    ]

    print("🚀 Running transaction bulk execution insertion loops...")
    try:
        # Erase previous transient data to keep runs completely repeatable
        cursor.execute("TRUNCATE TABLE reviews CASCADE;")
        
        insert_query = """
            INSERT INTO reviews (review_id, bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, identified_theme, source)
            VALUES %s ON CONFLICT (review_id) DO NOTHING;
        """
        execute_values(cursor, insert_query, review_tuples)
        
        # Commit transactional blocks safely
        conn.commit()
        print("✅ Data persistence completed seamlessly with full referential integrity constraints!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Core runtime error during bulk loading: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_data_to_postgres()