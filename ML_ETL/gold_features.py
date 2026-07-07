import os
from pathlib import Path
import duckdb
import pandas as pd

# Path routing configuration
SILVER_FILE = "./lakehouse/silver/cleaned_epilepsy_data.parquet"
GOLD_DIR = Path("./lakehouse/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = "./lakehouse/metadata.db"

def extract_gold_features():
    """
    Transforms clean Silver data into a normalized, numerical matrix 
    optimized for ML model ingestion (classification/regression tasks).
    """
    print("LOG [INFO]: Initiating Gold Layer feature engineering optimization...")
    
    if not os.path.exists(SILVER_FILE):
        raise FileNotFoundError(f"LOG [CRITICAL]: Missing cleaned Silver layer artifact at {SILVER_FILE}")
        
    # Read the secure Silver Parquet snapshot using DuckDB
    con = duckdb.connect()
    df_silver = con.execute(f"SELECT * FROM read_parquet('{SILVER_FILE}')").df()
    print(f"LOG [INFO]: Read {len(df_silver)} clean records from the Silver Layer.")

    # 1. Feature Engineering & Minimization
    # Isolate predictive inputs and drop structural hashes/metadata unnecessary for the model training step
    dropped_cols = [col for col in df_silver.columns if 'token' in col or 'id' in col]
    df_features = df_silver.drop(columns=dropped_cols, errors='ignore')
    print(f"LOG [SECURITY]: Stripped administrative/tracking attributes: {dropped_cols}")

    # 2. Automated Vector One-Hot Encoding for Complex Categorical Columns
    # Identify object/string variables (e.g., treatment types, gender identifiers, symptom types)
    categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"LOG [INFO]: Categorical features identified for numerical mapping: {categorical_cols}")
    
    # Transform strings to lightweight binary categorical indicator features
    df_gold = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True, dtype=int)

    # 3. Handle Remaining Empty Matrices cleanly for ML matrices
    numeric_cols = df_gold.select_dtypes(include=['number']).columns
    df_gold[numeric_cols] = df_gold[numeric_cols].fillna(0)

    # 4. Commit to Highly Optimized Gold Matrix partition
    target_file = GOLD_DIR / "ml_ready_features.parquet"
    df_gold.to_parquet(target_file, index=False, compression="snappy")
    print(f"LOG [INFO]: Production-ready Gold ML Feature Matrix committed: {target_file}")
    print(f"LOG [INFO]: Feature shape matrix summary: {df_gold.shape[1]} total ready features.")

    # 5. Lock Final Lineage State in Governance Ledger
    metadata_con = duckdb.connect(DATABASE_PATH)
    metadata_con.execute(
        "INSERT INTO data_lineage (layer, record_count, file_path) VALUES ('GOLD', ?, ?)",
        [len(df_gold), str(target_file)]
    )
    
    # Audit trail validation dump
    print("\n==================== CENTRAL LINEAGE REGISTER AUDIT ====================")
    audit_df = metadata_con.execute("SELECT * FROM data_lineage ORDER BY ingested_at ASC").df()
    print(audit_df.to_string(index=False))
    print("=========================================================================\n")
    metadata_con.close()

if __name__ == "__main__":
    extract_gold_features()