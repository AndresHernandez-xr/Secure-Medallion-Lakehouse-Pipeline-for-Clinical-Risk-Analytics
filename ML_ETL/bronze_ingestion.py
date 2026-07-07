import os
from pathlib import Path
from datasets import load_dataset
import duckdb

# Security & Infrastructure Configurations
BRONZE_DIR = Path("./lakehouse/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = "./lakehouse/metadata.db"

def ingest_raw_huggingface_data():
    """
    Extracts clinical/epilepsy synthetic data from Hugging Face 
    and loads it into the immutable Bronze layer as Parquet.
    """
    print("LOG [INFO]: Initiating connection to Hugging Face repository...")
    
    # 1. Extract: Fetching the synthetic epilepsy/neurological dataset
    try:
        dataset = load_dataset("electricsheepafrica/africa-synth-epilepsy-neurological-disorders-all", split="train")
        df_raw = dataset.to_pandas()
        print(f"LOG [INFO]: Successfully extracted {len(df_raw)} records.")
    except Exception as e:
        print(f"LOG [CRITICAL]: Extraction failed. Error: {str(e)}")
        raise

    # 2. Transform/Load: Standardize into an immutable Parquet snapshot
    target_file = BRONZE_DIR / "raw_epilepsy_data.parquet"
    
    # Write to Parquet using compression for optimization and immutability
    df_raw.to_parquet(target_file, index=False, compression="snappy")
    print(f"LOG [INFO]: Immutable Bronze snapshot written to: {target_file}")

    # 3. Governance: Initialize DuckDB catalog tracking for metadata audit
    con = duckdb.connect(DATABASE_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS data_lineage (
            layer VARCHAR,
            record_count INTEGER,
            file_path VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    con.execute(
        "INSERT INTO data_lineage (layer, record_count, file_path) VALUES ('BRONZE', ?, ?)",
        [len(df_raw), str(target_file)]
    )
    con.close()
    print("LOG [INFO]: Metadata lineage updated in central catalog.")

if __name__ == "__main__":
    ingest_raw_huggingface_data()