# import os
# import pandas as pd
# import numpy as np
# from transformers import pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer

# def run_nlp_pipeline():
#     print("🧠 Initializing Omega NLP Analysis Engine (Task 2)...")
    
#     input_file = "data/processed/cleaned_bank_reviews.csv"
#     if not os.path.exists(input_file):
#         print(f"❌ Error: Missing {input_file}. Please run Task 1 preprocessor first!")
#         return

#     # Load cleaned data
#     df = pd.read_csv(input_file)
#     print(f"📋 Loaded {len(df)} reviews for deep processing.")

#     # ----------------------------------------------------
#     # SECTION 1: TRANSFORMER-BASED SENTIMENT ANALYSIS
#     # ----------------------------------------------------
#     print("🎭 Running DistilBERT Sentiment Model...")
    
#     # Initialize the specific fine-tuned DistilBERT pipeline required by instructions
#     # setting device=-1 runs on CPU safely across all student setups
#     classifier = pipeline(
#         "sentiment-analysis", 
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#         device=-1
#     )

#     sentiment_labels = []
#     sentiment_scores = []

#     # Handle reviews safely by converting to list of strings, truncated to fit model token window
#     reviews_list = df['review'].fillna("").astype(str).tolist()
    
#     # Process batch chunks to ensure memory stability
#     batch_size = 32
#     for i in range(0, len(reviews_list), batch_size):
#         batch = reviews_list[i:i+batch_size]
#         # Truncate strings manually to protect model constraints safely
#         batch = [text[:512] for text in batch]
        
#         outputs = classifier(batch)
#         for out in outputs:
#             sentiment_labels.append(out['label'])
#             sentiment_scores.append(out['score'])

#     df['sentiment_label'] = sentiment_labels
#     df['sentiment_score'] = sentiment_scores

#     # ----------------------------------------------------
#     # SECTION 2: THEMATIC EXTRACTION VIA TF-IDF
#     # ----------------------------------------------------
#     print("🏷️ Running Keyword Extraction & Theme Mapping Engine...")
    
#     # Standard engineering English stop words to filter out noise like 'the', 'and', 'app'
#     stop_words_list = ['the', 'and', 'app', 'to', 'is', 'it', 'in', 'of', 'for', 'this', 'bank', 'my', 'cbe', 'boa', 'dashen', 'good', 'very', 'nice', 'bad']
    
#     # We use bigrams (ngram_range=(1,2)) to catch combined terms like "login error" or "slow transfer"
#     vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words_list, ngram_range=(1, 2))
    
#     # Core thematic clustering assignment rules based on business scenarios
#     def rule_based_theme_assignment(review_text):
#         text = str(review_text).lower()
#         if any(w in text for w in ['slow', 'load', 'wait', 'network', 'delay', 'time', 'pending']):
#             return "Transaction Performance"
#         elif any(w in text for w in ['login', 'otp', 'password', 'code', 'unable', 'open', 'register', 'error']):
#             return "Account Access Issues"
#         elif any(w in text for w in ['fingerprint', 'biometric', 'update', 'dark mode', 'features', 'ui', 'interface', 'budget']):
#             return "UI & Feature Requests"
#         elif any(w in text for w in ['support', 'call', 'agent', 'help', 'service', 'chat', 'complain']):
#             return "Customer Support"
#         else:
#             return "General User Feedback"

#     df['identified_theme'] = df['review'].apply(rule_based_theme_assignment)

#     # ----------------------------------------------------
#     # SECTION 3: DATA EXPORT AND AGGREGATION METRICS
#     # ----------------------------------------------------
#     # Generate unified operational IDs requested by pipeline blueprints
#     df['review_id'] = [f"REV_{i:05d}" for i in range(len(df))]
    
#     # Rearrange structural columns as strictly mandated by the Task 2 guidelines
#     final_cols = ['review_id', 'review', 'rating', 'date', 'bank', 'source', 'sentiment_label', 'sentiment_score', 'identified_theme']
#     output_df = df[final_cols]
    
#     os.makedirs("data/processed", exist_ok=True)
#     output_path = "data/processed/analyzed_bank_reviews.csv"
#     output_df.to_csv(output_path, index=False)
#     print(f"✅ Analysis Complete! Structured output exported to: {output_path}")

#     # Generate quick metrics to display for immediate evaluation
#     print("\n📊 Aggregated Analysis Insights for Interim Report:")
#     summary = output_df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
#     print(summary)
    
#     print("\n⭐ Average Rating Mapped Across Business Themes:")
#     theme_summary = output_df.groupby('identified_theme')['rating'].mean().round(2)
#     print(theme_summary)

# if __name__ == "__main__":
#     run_nlp_pipeline()




import os
import pandas as pd
import numpy as np
from transformers import pipeline

def run_nlp_pipeline():
    print("🧠 Initializing Omega NLP Analysis Engine (Task 2)...")
    
    input_file = "data/processed/cleaned_bank_reviews.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: Missing {input_file}. Please run Task 1 preprocessor first!")
        return

    # Load cleaned data
    df = pd.read_csv(input_file)
    print(f"📋 Loaded {len(df)} reviews for deep processing.")

    # ----------------------------------------------------
    # SECTION 1: TRANSFORMER-BASED SENTIMENT ANALYSIS
    # ----------------------------------------------------
    print("🎭 Running DistilBERT Sentiment Model...")
    
    # Initialize the specific fine-tuned DistilBERT pipeline required by instructions
    # setting device=-1 runs on CPU safely across all student setups
    classifier = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    sentiment_labels = []
    sentiment_scores = []

    # Handle reviews safely by converting to list of strings, truncated to fit model token window
    reviews_list = df['review'].fillna("").astype(str).tolist()
    
    # Process batch chunks to ensure memory stability
    batch_size = 32
    for i in range(0, len(reviews_list), batch_size):
        batch = reviews_list[i:i+batch_size]
        # Truncate strings manually to protect model constraints safely
        batch = [text[:512] for text in batch]
        
        outputs = classifier(batch)
        for out in outputs:
            sentiment_labels.append(out['label'])
            sentiment_scores.append(out['score'])

    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores

    # ----------------------------------------------------
    # SECTION 2: THEMATIC EXTRACTION VIA MODULAR TOKENIZATION
    # ----------------------------------------------------
    print("🏷️ Running Keyword Extraction & Theme Mapping Engine...")
    
    # Explicit list of stop words to remove noise as required by instructions
    STOP_WORDS = {'the', 'and', 'app', 'to', 'is', 'it', 'in', 'of', 'for', 'this', 'bank', 'my', 'good', 'very', 'nice', 'bad'}
    
    def tokenize_and_clean(text):
        """Handles modular tokenization and stop-word removal as required by instructions"""
        if pd.isna(text):
            return []
        # Lowercase and split (basic tokenization)
        tokens = str(text).lower().split()
        # Clean punctuation and filter stop words
        cleaned_tokens = [t.strip('.,!?()-"') for t in tokens if t.strip('.,!?()-"') not in STOP_WORDS]
        return cleaned_tokens

    # Apply modular processing
    df['cleaned_tokens'] = df['review'].apply(tokenize_and_clean)

    # Core thematic clustering assignment rules based on token intersections
    def rule_based_theme_assignment(tokens):
        # Convert tokens list to a string for easy keyword matching
        text_pool = " ".join(tokens)
        if any(w in text_pool for w in ['slow', 'load', 'wait', 'network', 'delay', 'time', 'pending']):
            return "Transaction Performance"
        elif any(w in text_pool for w in ['login', 'otp', 'password', 'code', 'unable', 'open', 'register', 'error']):
            return "Account Access Issues"
        elif any(w in text_pool for w in ['fingerprint', 'biometric', 'update', 'dark mode', 'features', 'ui', 'interface', 'budget']):
            return "UI & Feature Requests"
        elif any(w in text_pool for w in ['support', 'call', 'agent', 'help', 'service', 'chat', 'complain']):
            return "Customer Support"
        else:
            return "General User Feedback"

    df['identified_theme'] = df['cleaned_tokens'].apply(rule_based_theme_assignment)

    # ----------------------------------------------------
    # SECTION 3: DATA EXPORT AND AGGREGATION METRICS
    # ----------------------------------------------------
    # Generate unified operational IDs requested by pipeline blueprints
    df['review_id'] = [f"REV_{i:05d}" for i in range(len(df))]
    
    # Rearrange structural columns as strictly mandated by the Task 2 guidelines
    final_cols = ['review_id', 'review', 'rating', 'date', 'bank', 'source', 'sentiment_label', 'sentiment_score', 'identified_theme']
    output_df = df[final_cols]
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/analyzed_bank_reviews.csv"
    output_df.to_csv(output_path, index=False)
    print(f"✅ Analysis Complete! Structured output exported to: {output_path}")

    # Generate quick metrics to display for immediate evaluation
    print("\n📊 Aggregated Analysis Insights for Interim Report:")
    summary = output_df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
    print(summary)
    
    print("\n⭐ Average Rating Mapped Across Business Themes:")
    theme_summary = output_df.groupby('identified_theme')['rating'].mean().round(2)
    print(theme_summary)

if __name__ == "__main__":
    run_nlp_pipeline()