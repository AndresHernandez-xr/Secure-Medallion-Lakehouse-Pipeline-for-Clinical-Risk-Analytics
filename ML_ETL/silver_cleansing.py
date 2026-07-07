import os
import hmac
import hashlib
from pathlib import Path
import duckdb
import pandas as pd

# Path routing configuration
BRONZE_FILE = "./lakehouse/bronze/raw_epilepsy_data.parquet"
SILVER_DIR = Path("./lakehouse/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = "./lakehouse/metadata.db"

# Engineering Security Vault (In production, load this from AWS Secrets Manager / Azure Key Vault)
# The pepper adds cryptographic entropy to prevent dictionary attacks on clinical identifiers.
SECURITY_PEPPER = b"clinical-lakehouse-secure-pepper-2026-v1"

def tokenize_identifier(identifier: str) -> str:
    """
    Applies HMAC-SHA256 tokenization to transform plaintext identifiers 
    into secure, deterministic tokens for ML model grouping/linkage.
    """
    if pd.isna(identifier) or str(identifier).strip() == "":
        return "TOKEN_REDACTED_NULL"
    
    # Deterministic tokenization allows downstream models to group records without knowing identity
    secure_hash = hmac.new(SECURITY_PEPPER, str(identifier).encode('utf-8'), hashlib.sha256)
    return secure_hash.hexdigest()

def transform_bronze_to_silver():
    """
    Reads raw Bronze data, applies schema compliance rules, enforces de-identification
    cryptography, and outputs an optimized Silver Parquet dataset.
    """
    print("LOG [INFO]: Initializing Silver Layer transform process...")
    
    if not os.path.exists(BRONZE_FILE):
        raise FileNotFoundError(f"LOG [CRITICAL]: Missing raw Bronze snapshot at {BRONZE_FILE}")
        
    # Read the immutable Bronze Parquet snapshot via DuckDB for fast memory mapping
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{BRONZE_FILE}')").df()
    print(f"LOG [INFO]: Successfully pulled {len(df)} records from Bronze storage.")

    # 1. Identity Masking & Cryptographic Tokenization
    # Identifying tracking or ID columns in the dataset and transforming them securely
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'patient' in col.lower()]
    
    if id_cols:
        print(f"LOG [SECURITY]: Pseudonymizing sensitive identifiers: {id_cols}")
        for col in id_cols:
            df[col] = df[col].astype(str).apply(tokenize_identifier)
    else:
        # Fallback: In case the source lacks explicit IDs, synthesize and tokenize a secure cross-reference key
        print("LOG [SECURITY]: No explicit identifier columns detected. Creating a secure composite tracking key.")
        df['patient_secure_token'] = (df.index.astype(str) + "_salt_v1").apply(tokenize_identifier)

    # 2. Strict Schema Hardening & Type Validation
    # Let's cleanly cast data types, handle standard categorical keys, and normalize numeric fields
    for col in df.columns:
        # Standardize naming schema conventions
        df.rename(columns={col: col.strip().lower().replace(" ", "_")}, inplace=True)
    
    # Handle explicit clinical properties common to the Electric Sheep Africa datasets
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(-1).astype(int)
        
    # 3. Commit Compliant Dataset to Silver Partition Zone
    target_file = SILVER_DIR / "cleaned_epilepsy_data.parquet"
    df.to_parquet(target_file, index=False, compression="snappy")
    print(f"LOG [INFO]: Compliant Silver layer snapshot successfully committed: {target_file}")

    # 4. Governance Ledger Audit Update
    metadata_con = duckdb.connect(DATABASE_PATH)
    metadata_con.execute(
        "INSERT INTO data_lineage (layer, record_count, file_path) VALUES ('SILVER', ?, ?)",
        [len(df), str(target_file)]
    )
    metadata_con.close()
    print("LOG [INFO]: Governance catalog audit entries refreshed.")

if __name__ == "__main__":
    transform_bronze_to_silver()