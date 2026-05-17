import os
import pandas as pd
from google_play_scraper import Sort, reviews

# FINALIZED PRODUCTION PACKAGE REGISTRY MAP
BANKS = {
    "CBE": {
        "app_id": "prod.cbe.birr",            
        "name": "Commercial Bank of Ethiopia"
    },
    "BOA": {
        "app_id": "com.boa.boaMobileBanking", # UPDATED: Live production package identifier for BoA
        "name": "Bank of Abyssinia"
    },
    "Dashen": {
        "app_id": "com.dashen.dashensuperapp", 
        "name": "Dashen Bank"
    }
}

def scrape_bank_reviews():
    print("🚀 Running Advanced Web Scraping Engine with Realigned Package Map...")
    os.makedirs("data/raw", exist_ok=True)
    
    # Localization fallback matrix
    search_strategies = [
        {"lang": "en", "country": "us"},
        {"lang": "en", "country": "et"},
    ]
    
    for bank_key, bank_info in BANKS.items():
        print(f"\n=========================================")
        print(f"Targeting Institution: {bank_info['name']}")
        print(f"Active App ID Target:  {bank_info['app_id']}")
        print(f"=========================================")
        
        all_bank_reviews = []
        seen_review_ids = set()
        
        for strategy in search_strategies:
            if len(all_bank_reviews) >= 500:
                break
                
            print(f"Scanning Storefront (lang: '{strategy['lang']}', country: '{strategy['country']}')")
            
            try:
                result, _ = reviews(
                    bank_info['app_id'],
                    lang=strategy['lang'],
                    country=strategy['country'],
                    sort=Sort.NEWEST,
                    count=600  
                )
                
                if result:
                    new_records = 0
                    for rev in result:
                        if rev['reviewId'] not in seen_review_ids:
                            seen_review_ids.add(rev['reviewId'])
                            all_bank_reviews.append(rev)
                            new_records += 1
                    print(f"➡️ Captured {new_records} new unique reviews from this strategy pool.")
                else:
                    print(f"⚠️ Zero items matched for this specific strategy quadrant.")
                    
            except Exception as e:
                print(f"⚠️ Channel bypassed: App identifier restrictions or region lock.")
        
        print(f"🏁 Cumulative Reviews Isolated for {bank_key}: {len(all_bank_reviews)}")
        
        if len(all_bank_reviews) > 0:
            df_raw = pd.DataFrame(all_bank_reviews)
            df_raw['bank_name'] = bank_info['name']
            df_raw['source'] = "Google Play"
            
            output_path = f"data/raw/{bank_key}_raw_reviews.csv"
            df_raw.to_csv(output_path, index=False)
            print(f"✅ Successfully exported raw unstructured feed to: {output_path}")
        else:
            print(f"❌ CRITICAL ERROR: Database returned blank array for {bank_info['name']}.")

if __name__ == "__main__":
    scrape_bank_reviews()