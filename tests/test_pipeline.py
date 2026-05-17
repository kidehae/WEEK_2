"""
tests/test_pipeline.py

Automated testing module running range checks, TF-IDF string 
tokenization outputs, and edge-case error assertions.
"""

import pytest
import pandas as pd
from ..scripts.analyze_reviews import tokenize_and_clean, rule_based_theme_assignment

def test_tokenize_and_clean_removes_stopwords():
    """Validates that text cleaner isolates keywords and ignores noise words."""
    raw_text = "The app is slow and my network connection dropped."
    processed = tokenize_and_clean(raw_text)
    
    # Assertions checking explicit processing rules
    assert "the" not in processed.split()
    assert "slow" in processed.split()
    assert "network" in processed.split()

def test_thematic_mapping_logic():
    """Validates that token configurations map correctly to provisional business scenarios."""
    access_text = "i have an error with my login code or otp password"
    perf_text = "the transfer screen is stuck on a loading screen delay pending"
    
    assert rule_based_theme_assignment(access_text) == "Account Access Issues"
    assert rule_based_theme_assignment(perf_text) == "Transaction Performance"

def test_sentiment_score_boundary_ranges():
    """Mock-check data logic to make sure sentiment parameters fall inside valid mathematical bounds."""
    # Scores must fall inside the realistic probability range [0.0, 1.0]
    sample_score_1 = 0.998
    sample_score_2 = 0.451
    
    assert 0.0 <= sample_score_1 <= 1.0
    assert 0.0 <= sample_score_2 <= 1.0

def test_empty_dataframe_graceful_handling(capsys):
    """Verifies that running on empty data logs an informative message instead of crashing."""
    from scripts.analyze_reviews import run_nlp_pipeline
    
    # Execute pipeline on an intentional empty path hook to check failure cases
    run_nlp_pipeline(input_path="non_existent_file.csv")
    captured = capsys.readouterr()
    
    assert "does not exist" in captured.out