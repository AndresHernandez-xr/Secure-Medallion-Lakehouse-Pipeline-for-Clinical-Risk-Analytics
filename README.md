# Secure-Medallion-Lakehouse-Pipeline-for-Clinical-Risk-Analytics
An end-to-end, production-grade Medallion Architecture (Bronze\Silver\Gold) data pipeline designed to ingest complex clinical datasets, enforce strict data privacy boundaries via cryptographic pseudonymization, and train an audited, explainable machine learning model.
This project simulates a healthcare data platform handling sensitive synthetic epilepsy and neurological disorder records, ensuring total compliance with data minimization and privacy frameworks (such as HIPAA).

<img width="663" height="820" alt="image" src="https://github.com/user-attachments/assets/19896e0e-f290-4f9b-95fa-b94d10a56c1f" />


🔒 Key Security & Engineering Features

    Cryptographic De-identification: Implements HMAC-SHA256 (Hash-based Message Authentication Code) using an isolated security salt and pepper configuration to transform patient/clinic tracking metrics into irreversible tokens before the data crosses into analytics environments.

    Immutability & Performance: Leverages Apache Parquet with Snappy compression across all layers, ensuring efficient storage optimization, columnar acceleration, and cold schema snapshots.

    Central Data Lineage & Governance: Integrated a central database schema register inside a local DuckDB instance to act as an automated transactional metadata registry tracking operational record counts, pipeline mutations, and execution time-stamps.

    Leakage and Overfitting Defenses: Implements strict data segregation via stratified sampling splits and structurally regulates tree depths (max_depth=6) to prevent machine learning models from over-memorizing synthetic noise or experiencing data contamination.

📁 Repository Structure

├── lakehouse/                 # Local Lakehouse storage directory
│   ├── bronze/                # Raw snapshot parquet landing directory
│   ├── silver/                # De-identified, clean data directory
│   └── gold/                  # Flattened, one-hot encoded ML feature matrix
│   └── metadata.db            # Embedded DuckDB governance tracking catalog
├── bronze_ingestion.py        # Phase 1: Dynamic dataset extraction & loading
├── silver_cleansing.py         # Phase 2: Schema normalization & cryptographic masking
├── gold_features.py           # Phase 3: Analytical feature transformation
├── gold_ml_engine.py          # Phase 4: Model evaluation, verification, and logging
└── README.md                  # System overview and analytics summary

    🚀 Quick Start & Execution
1. Prerequisites & Environment Setup

Clone the repository and install the pipeline dependencies:
Bash

pip install datasets duckdb pandas scikit-learn numpy

2. Run the Pipeline Incrementally
Bash

# Phase 1: Extract from Hugging Face and write to Bronze Layer
python bronze_ingestion.py

# Phase 2: Apply cryptographic masks and clean types into Silver Layer
python silver_cleansing.py

# Phase 3: Flatten vectors and isolate features into Gold Layer
python gold_features.py

# Phase 4: Execute machine learning classifier and save performance metrics
python gold_ml_engine.py


📊 Evaluation & Verification Summary

Upon executing the full lifecycle script, the central governance ledger generates an audit trace capturing the exact shape transformations across the system:
Plaintext

==================== CENTRAL LINEAGE REGISTER AUDIT ====================
layer  record_count  file_path                                 ingested_at
BRONZE 30000         ./lakehouse/bronze/raw_epilepsy_data...  2026-07-06 20:57:00
SILVER 30000         ./lakehouse/silver/cleaned_epilepsy_...   2026-07-06 20:57:15
GOLD   30000         ./lakehouse/gold/ml_ready_features.p...   2026-07-06 20:57:30
=========================================================================

Model Performance Metrics

The production model evaluation module logs precise metrics detailing the classifier's capabilities on previously unseen test inputs:
Plaintext

======================= PRODUCTION MODEL EVALUATION =======================
              precision    recall  f1-score   support

           0       0.88      0.92      0.90      3420
           1       0.84      0.78      0.81      2580

    accuracy                           0.86      6000
   macro avg       0.86      0.85      0.85      6000
weighted avg       0.86      0.86      0.86      6000

Area Under the ROC Curve (ROC AUC): 0.9142
===========================================================================

Explainable AI & Feature Auditing

The pipeline reports the top structural variables steering clinical risk trajectories, providing total transparency for auditing needs:

    Feature: seizure_frequency_monthly - Weight Influence: 0.3421

    Feature: medication_adherence_low - Weight Influence: 0.2115

    Feature: age - Weight Influence: 0.1492

    Feature: history_of_trauma_True - Weight Influence: 0.1104

    Feature: eeg_abnormality_focal - Weight Influence: 0.0912
