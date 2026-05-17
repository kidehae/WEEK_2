import os
import glob
import pandas as pd

def preprocess_pipeline():
    print("🧹 Starting Data Preprocessing Pipeline...")
    raw_files = glob.glob("data/raw/*_raw_reviews.csv")
    
    if not raw_files:
        print("❌ Error: No raw CSV files found in data/raw/. Please run the scraper first.")
        return

    combined_list = []
    
    # Metrics tracking for final report documentation
    total_raw_rows = 0
    total_dropped_missing = 0
    total_dropped_duplicates = 0

    for file_path in raw_files:
        df = pd.read_csv(file_path)
        total_raw_rows += len(df)
        
        # 1. Handle missing values: drop rows missing vital data points
        # If there's no text ('content') or score ('score'), it's useless for sentiment analysis
        initial_count = len(df)
        df = df.dropna(subset=['content', 'score'])
        total_dropped_missing += (initial_count - len(df))
        
        # 2. Extract and Map only the essential columns required by Task 1 brief:
        # Expected: review, rating, date, bank, source
        df_mapped = pd.DataFrame({
            'review': df['content'].astype(str),
            'rating': df['score'].astype(int),
            'date': df['at'], # Raw timestamp column from scraper
            'bank': df['bank_name'],
            'source': df['source']
        })
        
        combined_list.append(df_mapped)

    # Merge all three bank dataframes together
    final_df = pd.concat(combined_list, ignore_index=True)
    
    # 3. Deduplication: Users frequently double-post identical reviews
    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset=['review', 'date', 'bank'])
    total_dropped_duplicates += (before_dedup - len(final_df))
    
    # 4. Date Normalization: Convert complex timestamps to standard YYYY-MM-DD string format
    final_df['date'] = pd.to_datetime(final_df['date']).dt.strftime('%Y-%m-%d')
    
    # Enforce strict output paths (data/ is protected by .gitignore)
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/cleaned_bank_reviews.csv"
    final_df.to_csv(output_file, index=False)
    
    # Print operational metrics critical for your KPI evaluation and interim report
    print("\n📈 Pipeline Execution Summary Statistics:")
    print(f"└── Total Raw Rows Gathered: {total_raw_rows}")
    print(f"└── Rows dropped due to missing data: {total_dropped_missing}")
    print(f"└── Rows dropped due to duplication: {total_dropped_duplicates}")
    print(f"└── Cleaned & Processed Rows remaining: {len(final_df)}")
    print(f"✨ Cleaned dataset successfully output to: {output_file}")

if __name__ == "__main__":
    preprocess_pipeline()