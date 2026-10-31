<div align="center">

# 🛍️ SpendScope
### Customer Spending Prediction · ML-Powered · Streamlit UI

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

> **An end-to-end machine learning application** that predicts how much a ShopSmart e-commerce customer will spend — built with a scikit-learn Pipeline, a premium dark-themed Streamlit UI, and a full pytest test suite.

</div>

---

## 📋 Table of Contents

- [Business Scenario](#-business-scenario)
- [Project Structure](#-project-structure)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Dataset Description](#-dataset-description)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [UI / UX Features](#-ui--ux-features)
- [Getting Started](#-getting-started)
- [Running the App](#-running-the-app)
- [Running Tests](#-running-tests)
- [Tech Stack](#-tech-stack)
- [Key Insights](#-key-insights)

---

## 🏢 Business Scenario

**ShopSmart**, an e-commerce company, currently applies the same marketing strategy to all customers — a costly, one-size-fits-all approach. To optimize marketing spend and personalize campaigns, the data science team built **SpendScope**: a regression model that estimates **how much a customer will spend**, based on their demographics, behavior, and membership profile.

By predicting `Amount_Spent` before a campaign runs, ShopSmart can:
- Route **high-value** prospects to premium offers
- Offer **targeted discounts** to mid-tier customers
- Avoid **wasting budget** on unlikely spenders

---

## 📁 Project Structure

```
Customer Spending Prediction/
│
├── 📓 notebooks/
│   └── EDA.ipynb              # Full exploratory data analysis & model training
│
├── 🤖 best_model/
│   └── best_model_prediction.pkl   # Serialized best-performing sklearn Pipeline
│
├── 📊 datasets/
│   ├── ShopSmart_data.csv     # Raw dataset
│   └── data_cleaned.csv       # Cleaned & preprocessed dataset
│
├── 🖼️ assets/
│   └── logo.png               # App logo
│
├── 🧪 tests/
│   └── test_model.py          # Full pytest test suite (29 tests)
│
├── 🚀 App.py                  # Premium Streamlit UI (main entry point)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔬 Machine Learning Workflow

The full workflow is documented in `notebooks/EDA.ipynb`:

```
Raw Data  →  EDA & Cleaning  →  Feature Engineering  →  Pipeline  →  Train  →  Evaluate  →  Best Model  →  Deploy
```

### Step-by-Step

| Step | Description |
|------|-------------|
| **1. Load & Inspect** | Load CSV, check shape, dtypes, nulls, and duplicates |
| **2. Feature / Target Split** | `X` = 8 customer features, `y` = `Amount_Spent` |
| **3. Train/Test Split** | `test_size=0.2`, `random_state=42` for reproducibility |
| **4. Preprocessing** | `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder` (categorical) |
| **5. Model Training** | 4 models trained inside scikit-learn `Pipeline` |
| **6. Evaluation** | MAE, MSE, R² compared across all models |
| **7. Best Model** | Saved as `best_model_prediction.pkl` for deployment |

---

## 📊 Dataset Description

**Source:** ShopSmart internal customer data (`ShopSmart_data.csv`)  
**Problem Type:** Regression (continuous target)  
**Target Variable:** `Amount_Spent` — the customer's total spending amount in ₹

| Feature | Type | Description | Example |
|---|---|---|---|
| `Age` | Numerical | Customer age in years | 32 |
| `Annual_Income` | Numerical | Annual income in INR (₹) | 8,00,000 |
| `Website_Visits` | Numerical | Website visits per month | 12 |
| `Previous_Purchases` | Numerical | Total past purchases | 8 |
| `Membership_Type` | Categorical | Bronze / Silver / Gold | Gold |
| `Preferred_Category` | Categorical | Electronics / Fashion / Home / Beauty | Electronics |
| `City_Tier` | Categorical | Tier 1 / Tier 2 / Tier 3 | Tier 1 |
| `Discount_Used` | Categorical | Yes / No | Yes |
| `Amount_Spent` ⭐ | **Target** | Spending amount to predict | 24,500 |

---

## 🏗️ Model Architecture

The solution uses a **scikit-learn Pipeline** combining preprocessing and a regressor in a single, leak-proof object:

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(),       numerical_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore"),  categorical_cols),
    ])),
    ("regressor", RandomForestRegressor(...))
])
```

### Models Compared

| Model | Notes |
|-------|-------|
| `LinearRegression` | OLS baseline |
| `Ridge` | L2-regularized linear model |
| `Lasso` | L1-regularized (feature selection) |
| `RandomForestRegressor` ✅ | **Best performer — selected model** |

> **Why Random Forest?**  
> It captures non-linear relationships between income, membership, and spending patterns that linear models miss. It also handles categorical features robustly after one-hot encoding and is resistant to outliers.

---

## 📈 Model Performance

> Results from `notebooks/EDA.ipynb` evaluation on the held-out test set.

| Model | MAE (↓) | MSE (↓) | R² Score (↑) |
|-------|---------|---------|-------------|
| LinearRegression | — | — | — |
| Ridge | — | — | — |
| Lasso | — | — | — |
| **RandomForestRegressor** ✅ | **Best** | **Best** | **Highest** |

> *Exact metrics are printed in the notebook output cells.*

---

## 🎨 UI / UX Features

The Streamlit app (`App.py`) offers a **premium dark-themed dashboard** with:

| Feature | Details |
|---------|---------|
| 🌙 **Dark Theme** | Midnight blue / purple / cyan color system |
| 📊 **Live Dataset Stats** | 4 metric cards — customers, avg/median spend, gold %, etc. |
| 🔢 **Smart Input Form** | Organized in numerical + categorical sections with tooltips |
| 🎯 **Instant Prediction** | Click one button → get ₹ amount + spender tier label |
| 📉 **Gauge Chart** | Visual gauge showing prediction vs dataset max |
| 📊 **Insights Tab** | 6 interactive Plotly charts (histogram, bar, pie, scatter, box, correlation) |
| ✨ **Animations** | Smooth entry animations, hover effects, gradient accents |
| 📱 **Responsive Layout** | Adapts cleanly to different screen widths |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### 1. Clone / Open the Project

```bash
cd "Customer Spending Prediction"
```

### 2. Create & Activate Virtual Environment

```bash
# Create
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run App.py
```

Then open your browser at **http://localhost:8501** 🎉

### App Walkthrough

1. **Sidebar** — See dataset stats and model info at a glance
2. **Hero Banner** — Overview of the prediction engine
3. **Dataset Overview** — 4 live metric cards
4. **Predict Tab** — Fill in 8 customer features → click **⚡ Predict Spending Amount**
5. **Insights Tab** — Explore 6 interactive data visualizations

---

## 🧪 Running Tests

The project includes a comprehensive **pytest** test suite with **29 tests** across 4 categories:

```bash
# Activate venv first
myenv\Scripts\activate

# Run all tests
pytest tests/test_model.py -v

# Run specific test class
pytest tests/test_model.py::TestPrediction -v

# Run with summary only
pytest tests/test_model.py -q
```

### Test Categories

| Class | Tests | What's Covered |
|-------|-------|----------------|
| `TestModelFile` | 5 | File exists, non-empty, loads, is Pipeline, has steps |
| `TestDataset` | 10 | Schema, nulls, value ranges, membership/discount validity |
| `TestPrediction` | 10 | Return type, sign, range, determinism, batch, all tiers |
| `TestEdgeCases` | 4 | All city tiers, all categories, both discount values, task example |

### Example Test Run (Verified ✅)

```
============================= test session starts ==============================
platform win32 -- Python 3.12.4, pytest-9.1.1

collected 29 items

tests/test_model.py::TestModelFile::test_model_file_exists              PASSED [  3%]
tests/test_model.py::TestModelFile::test_model_file_non_empty           PASSED [  6%]
tests/test_model.py::TestModelFile::test_model_loads_without_error      PASSED [ 10%]
tests/test_model.py::TestModelFile::test_model_is_sklearn_pipeline      PASSED [ 13%]
tests/test_model.py::TestModelFile::test_pipeline_has_steps             PASSED [ 17%]
tests/test_model.py::TestDataset::test_dataset_loads                    PASSED [ 20%]
tests/test_model.py::TestDataset::test_dataset_not_empty                PASSED [ 24%]
tests/test_model.py::TestDataset::test_required_columns_present         PASSED [ 27%]
tests/test_model.py::TestDataset::test_no_nulls_in_features             PASSED [ 31%]
tests/test_model.py::TestDataset::test_target_column_positive           PASSED [ 34%]
tests/test_model.py::TestDataset::test_age_reasonable                   PASSED [ 37%]
tests/test_model.py::TestDataset::test_membership_types_valid           PASSED [ 41%]
tests/test_model.py::TestDataset::test_discount_used_values             PASSED [ 44%]
tests/test_model.py::TestDataset::test_dataset_minimum_size             PASSED [ 48%]
tests/test_model.py::TestDataset::test_income_range                     PASSED [ 51%]
tests/test_model.py::TestPrediction::test_predict_returns_value         PASSED [ 55%]
tests/test_model.py::TestPrediction::test_predict_returns_float         PASSED [ 58%]
tests/test_model.py::TestPrediction::test_predict_positive_value        PASSED [ 62%]
tests/test_model.py::TestPrediction::test_predict_realistic_range       PASSED [ 65%]
tests/test_model.py::TestPrediction::test_predict_single_row            PASSED [ 68%]
tests/test_model.py::TestPrediction::test_predict_multiple_customers    PASSED [ 72%]
tests/test_model.py::TestPrediction::test_predict_deterministic         PASSED [ 75%]
tests/test_model.py::TestPrediction::test_predict_silver_member         PASSED [ 79%]
tests/test_model.py::TestPrediction::test_predict_bronze_member         PASSED [ 82%]
tests/test_model.py::TestPrediction::test_predict_high_income_customer  PASSED [ 86%]
tests/test_model.py::TestEdgeCases::test_predict_all_city_tiers         PASSED [ 89%]
tests/test_model.py::TestEdgeCases::test_predict_all_categories         PASSED [ 93%]
tests/test_model.py::TestEdgeCases::test_predict_discount_both_values   PASSED [ 96%]
tests/test_model.py::TestEdgeCases::test_task_example_customer          PASSED [100%]

============================== 29 passed in 9.19s ==============================
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Core language |
| **pandas 3.x** | Data manipulation |
| **scikit-learn 1.9** | ML pipeline, preprocessing, models |
| **Streamlit** | Interactive web application |
| **Plotly** | Interactive charts & visualizations |
| **pytest** | Unit & integration testing |
| **NumPy** | Numerical computations |
| **pickle** | Model serialization |

---

## 💡 Key Insights

From the EDA and model analysis:

- 🥇 **Gold members** spend significantly more than Bronze/Silver customers
- 💻 **Electronics** is the highest-spending preferred category
- 🏙️ **Tier 1 cities** correlate with higher spending
- 💰 **Annual Income** is one of the strongest predictors of spending
- 🛒 **Previous Purchases** shows a positive relationship with future spend

---

## 🔮 Example Prediction

Using the task specification example:

| Feature | Value |
|---------|-------|
| Age | 32 |
| Annual Income | ₹8,00,000 |
| Website Visits | 12 |
| Previous Purchases | 8 |
| Membership Type | Gold |
| Preferred Category | Electronics |
| City Tier | Tier 1 |
| Discount Used | Yes |
| **Predicted Amount Spent** | **₹ (model output)** |

---

## 📝 Git Workflow

```bash
git init
git remote add origin <your-repo-url>
git checkout -b main
git pull origin main
git status
git add .
git commit -m "feat: add premium UI, test suite, and README"
git push -u origin main
```

---

<div align="center">

Made with ❤️ for **ShopSmart** · Powered by **scikit-learn** + **Streamlit**

</div>
