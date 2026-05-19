"""
scripts/generate_plots.py

Connects to the bank_reviews PostgreSQL database, extracts analyzed metrics,
and generates production-ready Seaborn visualizations for the final report.
"""

import os
import sys
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Database configuration
DB_HOST = "localhost"
DB_NAME = "bank_reviews"
DB_USER = "postgres"
DB_PASS = "postgresql"  # Update with your password
DB_PORT = "5432"

def generate_analytics_plots():
    print("🔌 Connecting to database to extract visualization metrics...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
        )
        query = """
            SELECT r.review_id, b.bank_name, r.rating, r.sentiment_label, r.identified_theme 
            FROM reviews r
            JOIN banks b ON r.bank_id = b.bank_id;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

    if df.empty:
        print("⚠️ No data found in the database. Run the database loader script first.")
        return

    # Set styling parameters for academic/consulting clean layout
    sns.set_theme(style="whitegrid")
    os.makedirs("notebooks", exist_ok=True)  # Storing plots safely inside project structure

    # ----------------------------------------------------
    # PLOT 1: SENTIMENT DISTRIBUTION BY BANK (Stacked/Grouped Bar)
    # ----------------------------------------------------
    print("📊 Generating Plot 1: Sentiment Distribution across Banks...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="bank_name", hue="sentiment_label", palette={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"})
    plt.title("Sentiment Classification Breakdown across Ethiopian Banking Apps", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Financial Institution", fontsize=12)
    plt.ylabel("Total Count of Customer Reviews", fontsize=12)
    plt.legend(title="Sentiment Category")
    plt.tight_layout()
    plt.savefig("notebooks/sentiment_distribution.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # PLOT 2: RATING DISTRIBUTION PER BANK (Boxplot)
    # ----------------------------------------------------
    print("📦 Generating Plot 2: Rating Spread Boxplot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="bank_name", y="rating", palette="Blues", width=0.5)
    plt.title("Distribution of App Star Ratings (1-5) by Institution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Financial Institution", fontsize=12)
    plt.ylabel("Star Rating Value", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/rating_boxplots.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # PLOT 3: THEME FREQUENCY COMPARISON (Horizontal Bar)
    # ----------------------------------------------------
    print("🏷️ Generating Plot 3: Dominant Business Theme Frequencies...")
    plt.figure(figsize=(11, 7))
    theme_counts = df.groupby(["identified_theme", "bank_name"]).size().reset_index(name="count")
    sns.barplot(data=theme_counts, y="identified_theme", x="count", hue="bank_name", palette="viridis")
    plt.title("Frequency Distribution of Business Themes across Customer Feedback", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Volume of Reviews Mapped", fontsize=12)
    plt.ylabel("Identified Operational Business Theme", fontsize=12)
    plt.legend(title="Bank Entity Location")
    plt.tight_layout()
    plt.savefig("notebooks/theme_frequencies.png", dpi=300)
    plt.close()

    print("✅ All plots generated successfully and persisted to the 'notebooks/' folder!")

if __name__ == "__main__":
    generate_analytics_plots()