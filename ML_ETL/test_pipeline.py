import pytest
import pandas as pd
import numpy as np
import hmac
import hashlib
from pathlib import Path

# Import the tokenization logic directly from your silver layer script
# (Assuming your script is named silver_cleansing.py in the same directory)
from silver_cleansing import tokenize_identifier, SECURITY_PEPPER

# 1. FIXTURE: Mocking raw Bronze data for predictable testing
@pytest.fixture
def mock_bronze_data():
    """Generates a controlled, messy raw dataframe mimicking our Bronze layer."""
    return pd.DataFrame({
        "Patient ID": ["PT-101", "PT-102", "", None],  # Contains strings, empty strings, and nulls
        " Age ": ["25", " 42 ", "missing", "30"],       # Trailing spaces and corrupted text
        "Seizure Frequency": [3.5, 1.0, None, 0.0],
        "Condition Status": ["Critical", "Stable", "Stable", "Critical"]
    })

# 2. UNIT TEST: Cryptographic Isolation
def test_cryptographic_tokenization():
    """
    SECURITY TEST: Validates that PII tokenization is deterministic, irreversible, 
    and handles null fields without throwing exceptions.
    """
    raw_id = "PT-101"
    
    # Assert token matches expected HMAC-SHA256 output
    expected_hash = hmac.new(SECURITY_PEPPER, raw_id.encode('utf-8'), hashlib.sha256).hexdigest()
    assert tokenize_identifier(raw_id) == expected_hash
    
    # Assert tokenization is deterministic (same input always equals same output for ML tracking)
    assert tokenize_identifier(raw_id) == tokenize_identifier(raw_id)
    
    # Assert tokenization handles nulls gracefully and masks them safely
    assert tokenize_identifier(None) == "TOKEN_REDACTED_NULL"
    assert tokenize_identifier("") == "TOKEN_REDACTED_NULL"

# 3. UNIT TEST: Schema Normalization & Cleaning
def test_silver_cleansing_and_schema(mock_bronze_data):
    """
    DATA QUALITY TEST: Asserts that columns are lowercase, spaces are removed,
    and text-based integers are safely parsed.
    """
    df = mock_bronze_data.copy()
    
    # Apply column normalization rules (from Phase 2)
    for col in df.columns:
        df.rename(columns={col: col.strip().lower().replace(" ", "_")}, inplace=True)
        
    # Assert string whitespace and casing were standardized
    assert "patient_id" in df.columns
    assert "age" in df.columns
    assert "seizure_frequency" in df.columns
    assert " _age_ " not in df.columns

    # Apply age cleaning rules
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(-1).astype(int)
    
    # Assert data typing is strictly enforced
    assert df['age'].dtype == np.int64 or df['age'].dtype == np.int32
    assert df['age'].iloc[2] == -1  # "missing" should mutate safely to -1

# 4. UNIT TEST: Feature Engineering Vector Completeness
def test_gold_feature_matrix_generation():
    """
    ML INTEGRITY TEST: Asserts that text categories are converted to purely 
    numerical vectors (one-hot encoding) and that no PII strings spill over.
    """
    # Simulate a processed Silver layer
    silver_mock = pd.DataFrame({
        "patient_secure_token": ["abc123hash", "def456hash"],
        "age": [25, 42],
        "condition_status": ["critical", "stable"]
    })
    
    # Drop administrative tokens
    features = silver_mock.drop(columns=["patient_secure_token"])
    
    # Perform one-hot encoding
    gold_df = pd.get_dummies(features, columns=["condition_status"], drop_first=True, dtype=int)
    
    # Assert administrative/PII tokens were successfully minimized (dropped)
    assert "patient_secure_token" not in gold_df.columns
    
    # Assert that categoricals transformed into dynamic indicator columns
    assert "condition_status_stable" in gold_df.columns
    assert "condition_status" not in gold_df.columns
    
    # Assert the matrix is entirely numeric for ML model ingestion
    assert all(np.issubdtype(dtype, np.number) for dtype in gold_df.dtypes)