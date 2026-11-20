# Parcel Delivery Days Prediction (Lucknow Hub)

**Course:** SCSE3040 MLOps  
**Problem ID:** MP012  
**Group ID:** G191  
**Batch:** 28  
**Faculty:** Saurabh Shrivastava  
**Repository:** [https://github.com/ShauriyaDeveloper1/Parcel-Delivery-Days-Lucknow-](https://github.com/ShauriyaDeveloper1/Parcel-Delivery-Days-Lucknow-)

---

## 1. Project Overview & Problem Description

The **Lucknow Speed Parcel Hub** operates a regional delivery network handling high parcel volumes across northern India. Due to complex routing lanes, variable hub congestion, and geographic disparities, promised delivery timelines often fail to match actual customer delivery days. Inaccurate estimates degrade customer trust and lead to elevated customer support inquiries.

The objective of this project is to design, implement, and deploy an end-to-end production MLOps pipeline that predicts delivery duration (`delivery_days`) for each parcel at the time of booking. By providing honest, data-driven delivery estimates, the hub can set realistic customer expectations, detect operational bottlenecks, and monitor performance across regional tiers.

### Prediction Target & Metric

- **Target Variable:** `delivery_days` (continuous numeric value representing elapsed days from parcel booking to customer delivery).
- **Task Type:** Supervised Regression.
- **Evaluation Metric:** **Mean Absolute Error (MAE)** — measuring the average absolute prediction discrepancy in days.

---

## 2. Group Members & Roles

| Enrollment No. | Student Name | Role | Responsibilities |
|---|---|---|---|
| **S24CSEU0409** | **SHAURIYAGARG** (You) | Data Lead | Git repository setup, DVC initialization, raw dataset versioning, exploratory data investigation |
| **S24CSEU0442** | **SahajKaur** | Model Lead | Data cleaning pipeline, feature engineering, model training v1/v2, MLflow tracking and model registry |
| **S24CSEU0393** | **SANCHITSIROHI** | Service Lead | FastAPI inference microservice, schema validation, Pytest suite, Docker containerization |
| **S24CSEU0488** | **Saurabhi** | Ops Lead | GitHub Actions CI pipeline, Kubernetes manifests, structured JSON logging, drift monitoring (PSI) |

---

## 3. Dataset Description

The project utilizes three distinct dataset splits provided by the client:

| File | Rows | Period | Description | Tracking Mechanism |
|---|---|---|---|---|
| `data/raw/train.csv` | ~1,000 | Jan – Jun 2026 | Historical training dataset containing operational noise, formatting inconsistencies, and potential recording anomalies. | Tracked via **DVC** (`train.csv.dvc`) |
| `data/raw/later.csv` | 300 | Jul – Aug 2026 | Subsequent clean operational records arriving post-deployment, used for drift evaluation and retraining. | Tracked via **DVC** (`later.csv.dvc`) |
| `data/raw/holdout_features.csv` | 250 | Sep 2026 | Evaluation holdout partition containing feature inputs without target labels. Used for the final benchmark. | Tracked via **Git** |

### Feature Schema

| Column | Type | Unit | Description |
|---|---|---|---|
| `record_id` | String | Identifier | Unique parcel tracking code |
| `date` | Date | YYYY-MM-DD | Booking date of the parcel |
| `distance_km` | Float | Kilometers | Transit distance to destination |
| `weight_kg` | Float | Kilograms | Measured parcel weight |
| `hub_backlog` | Integer | Count | Backlogged parcels waiting at Lucknow hub on booking day |
| `pin_code_tier` | String / Int | Category | Delivery zone tier classification |
| `declared_value_rs` | Float | INR (₹) | Declared monetary value of parcel contents |
| `transfers` | Integer | Count | Total intermediate hub-to-hub transfers |
| `service` | String | Category | Service level requested (`standard`, `express`, `economy`) |
| `destination` | String | Category | Destination area type (`Metro`, `Tier-2`, `Rural`) |
| `delivery_days` | Float / Int | Days | **Target:** Elapsed transit time in days from booking to delivery |

---

## 4. Technology Stack

- **Core Language:** Python 3.12
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Data & Pipeline Versioning:** `DVC` (Data Version Control)
- **Experiment Tracking & Model Registry:** `MLflow` (SQLite backend `sqlite:///mlflow.db`)
- **Inference API:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Automated Testing:** `pytest`, `httpx` (TestClient)
- **Containerization:** `Docker` (Alpine/Debian-slim base)
- **Continuous Integration:** `GitHub Actions`
- **Orchestration & Deployment:** `Kubernetes` (Deployment and Service manifests)

---

## 5. End-to-End MLOps Workflow

The project follows a phased lifecycle aligned with industry best practices:

```
[Raw Data] -> [DVC Tracking] -> [Data Cleaning & Validation] -> [MLflow Experimentation]
                                                                        |
                                                                        v
[Kubernetes Manifests] <- [Docker Container] <- [FastAPI Microservice] <- [Champion Model Registry]
                                                                        |
                                                                        v
                                                          [PSI Drift & Group Monitoring]
```

### Phased Execution Stages

1. **Stage 1 — Repository & Data Versioning Setup (Current Stage):**
   - Initialize Git repository and link to the public GitHub remote.
   - Configure project directory structure, dependencies (`requirements.txt`), and parameters (`params.yaml`).
   - Initialize DVC and track original raw CSVs (`train.csv`, `later.csv`) with pointer metadata.
   - Investigate raw data issues for documentation in `findings.yaml`.
2. **Stage 2 — Data Cleaning, Model Development & MLflow Tracking (Planned):**
   - Implement data cleaning modules in `src/parcel_ml/`.
   - Train baseline and candidate regression models logged to local SQLite MLflow backend.
   - Register champion model with the `@champion` alias and implement automated test suites.
3. **Stage 3 — Inference Service, Packaging & CI/CD (Planned):**
   - Develop FastAPI prediction endpoints (`POST /predict`, `GET /health`) with Pydantic validation rules:
     - Reject requests where `transfers > 10` (HTTP 422).
     - Reject requests where `declared_value_rs < 50` (HTTP 422).
   - Dockerize the application and construct Kubernetes deployment manifests.
   - Set up GitHub Actions CI workflow to run test suites on push.
4. **Stage 4 — Drift Monitoring, Retraining & Final Predictions (Planned):**
   - Compute Population Stability Index (PSI) per numeric column on `later.csv`.
   - Assess demographic performance across destination segments (`Metro`, `Tier-2`, `Rural`).
   - Retrain model v2 on combined data and evaluate against promotion threshold (≥8% MAE improvement).
   - Generate final batch predictions on `data/raw/holdout_features.csv`.

---

## 6. Project Directory Structure

```
G191_MP012/
├── .dvc/                             # DVC internal configuration and tracking state
│   ├── .gitignore
│   └── config
├── .dvcignore                        # DVC ignore rules
├── data/
│   └── raw/
│       ├── .gitignore                # Git ignore ensuring large CSVs are excluded
│       ├── train.csv.dvc             # DVC pointer for historical training dataset
│       ├── later.csv.dvc             # DVC pointer for post-launch dataset
│       └── holdout_features.csv      # Unlabeled holdout feature set (tracked in Git)
├── src/
│   └── parcel_ml/                    # Core Python package for cleaning, modeling, and inference
│       └── __init__.py
├── tests/                            # Pytest test suite directory
│   └── .gitkeep
├── findings.yaml                     # Official findings template for data anomalies and metrics
├── params.yaml                       # Pipeline configuration and hyperparameters
├── requirements.txt                  # Pinned Python package dependencies
├── PROBLEM.md                        # Official problem specification
├── STUDENT_HANDBOOK.md               # Course guidelines and evaluation rubric
├── GROUP.md                          # Group allocation and member details
└── README.md                         # Project documentation and instructions
```

---

## 7. Data Versioning & Setup Guide for Collaborators

Large data files (`train.csv` and `later.csv`) are managed using **DVC** and are intentionally excluded from the Git remote repository to prevent repository bloat and maintain strict version control integrity.

### How Teammates Access and Verify Data

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShauriyaDeveloper1/Parcel-Delivery-Days-Lucknow-.git
   cd Parcel-Delivery-Days-Lucknow-
   ```

2. **Locate Your Group Archive:**
   - Every group member has been provided with the official `G191_MP012.zip` distribution.

3. **Populate Raw Data Directory:**
   - Extract `train.csv` and `later.csv` from your `G191_MP012.zip` directly into the `data/raw/` directory.
   - *Important:* Never open or save these CSV files in Microsoft Excel or external spreadsheet software, as auto-formatting alters file timestamps and cryptographic hashes.

4. **Verify DVC Synchronization:**
   - Ensure DVC is installed (`pip install dvc==3.67.1`).
   - Run the status check:
     ```bash
     dvc status
     ```
   - If the original unedited CSV files are present in `data/raw/`, DVC will report:
     ```
     Data and pipelines are up to date.
     ```

5. **Verify File Integrity (SHA-256 Checksums):**
   You can verify the cryptographic hashes of your local files using Python:
   ```bash
   python -c "import hashlib; print('train.csv:', hashlib.sha256(open('data/raw/train.csv', 'rb').read()).hexdigest())"
   python -c "import hashlib; print('later.csv:', hashlib.sha256(open('data/raw/later.csv', 'rb').read()).hexdigest())"
   ```
   Expected SHA-256 values:
   - `train.csv`: `832feb9a92f32dc03087beaf6f4b175af9ed0f8493bada633218b43e4299170a`
   - `later.csv`: `cb42e3982b958dd40b49f76b50733850a40dcebbd3b426efdbcf823d9f0a64da`
   - `holdout_features.csv`: `fb1eb70796be734c8760d7b5fd73e91278c0dc755cd833b53b51e9477bce7f94`
