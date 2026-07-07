import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Path routing configuration
GOLD_FILE = "./lakehouse/gold/ml_ready_features.parquet"
DATABASE_PATH = "./lakehouse/metadata.db"

def run_production_ml_pipeline():
    """
    Ingests the secure Gold Feature Matrix, segregates target vectors,
    trains an audited Random Forest classifier, and evaluates production readiness.
    """
    print("LOG [INFO]: Initializing Production Machine Learning Engine...")
    
    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(f"LOG [CRITICAL]: Missing Gold Feature Matrix at {GOLD_FILE}")
        
    # Read our engineered Gold Feature Matrix
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{GOLD_FILE}')").df()
    con.close()
    
    # Dynamically select a clinical target variable from the synthetic dataset
    # Look for status, type, or diagnosis flags; fallback to the last generated binary flag if needed
    potential_targets = [col for col in df.columns if 'status' in col or 'severity' in col or 'condition' in col]
    
    if potential_targets:
        target_col = potential_targets[0]
    else:
        # Fallback to the final column as a synthetic target vector if strict diagnostic flags aren't matched
        target_col = df.columns[-1]
        
    print(f"LOG [INFO]: Targeted classification vector established: '{target_col}'")
    
    # 1. Segregate Features from Target Vector
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # If the target vector is a continuous metric or multi-class text, binarize for a clear risk classification profile
    if y.dtype == 'object' or len(np.unique(y)) > 2:
        # Cast to binary based on the most common class threshold or factorized state
        y = (y == y.mode()[0]).astype(int)
        print(f"LOG [INFO]: Normalized target vector into a binary risk threshold.")

    # 2. Strict Train-Test Separation (Preventing Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"LOG [INFO]: Train-Test split locked down. Training shape: {X_train.shape}, Evaluation shape: {X_test.shape}")

    # 3. Secure Model Execution & Auditing
    print("LOG [INFO]: Training audited Random Forest Classifier model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Evaluation & Production Metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n======================= PRODUCTION MODEL EVALUATION =======================")
    print(classification_report(y_test, y_pred))
    try:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"Area Under the ROC Curve (ROC AUC): {auc_score:.4f}")
    except Exception:
        pass
    print("===========================================================================\n")

    # 5. Governance: Feature Importance & Explainable AI Audit
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5]  # View the top 5 driving indicators
    
    print("LOG [GOVERNANCE]: Top 5 Features Driving Clinical Risk Assessment:")
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1}. Feature: '{X.columns[idx]}' - Weight Influence: {importances[idx]:.4f}")

    # Log model training execution meta-parameters to central database for security auditing
    metadata_con = duckdb.connect(DATABASE_PATH)
    metadata_con.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_name VARCHAR,
            features_used INTEGER,
            evaluated_records INTEGER,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    metadata_con.execute(
        "INSERT INTO model_registry (model_name, features_used, evaluated_records) VALUES ('Clinical_Risk_RF', ?, ?)",
        [X.shape[1], len(X_test)]
    )
    metadata_con.close()
    print("\nLOG [INFO]: Model registry audit trail finalized.")

if __name__ == "__main__":
    run_production_ml_pipeline()