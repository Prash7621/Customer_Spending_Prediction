"""
╔══════════════════════════════════════════════════════════╗
║          SpendScope — Test Suite                         ║
║          Testing: Model, Pipeline, Data, Inference       ║
╚══════════════════════════════════════════════════════════╝

Run:
    pytest tests/test_model.py -v
"""

import os
import sys
import pickle
import pytest
import numpy as np
import pandas as pd

# ── Resolve project root ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL_PATH   = os.path.join(ROOT, "best_model", "best_model_prediction.pkl")
DATASET_PATH = os.path.join(ROOT, "datasets", "data_cleaned.csv")

# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    """Load the trained sklearn pipeline once for the whole test session."""
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def dataset():
    """Load the cleaned dataset once for the whole test session."""
    assert os.path.exists(DATASET_PATH), f"Dataset not found: {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


@pytest.fixture
def sample_customer():
    """A typical valid customer profile DataFrame."""
    return pd.DataFrame({
        "Age":                [32],
        "Annual_Income":      [800_000],
        "Website_Visits":     [12],
        "Previous_Purchases": [8],
        "Membership_Type":    ["Gold"],
        "Preferred_Category": ["Electronics"],
        "City_Tier":          ["Tier 1"],
        "Discount_Used":      ["Yes"],
    })


# ══════════════════════════════════════════════════════════════════
#  1. MODEL FILE TESTS
# ══════════════════════════════════════════════════════════════════

class TestModelFile:
    def test_model_file_exists(self):
        """Model pickle file must exist in best_model directory."""
        assert os.path.exists(MODEL_PATH), "Model file is missing!"

    def test_model_file_non_empty(self):
        """Model file should not be empty (corrupt)."""
        assert os.path.getsize(MODEL_PATH) > 0, "Model file is empty!"

    def test_model_loads_without_error(self, pipeline):
        """Model must be loadable via pickle without raising."""
        assert pipeline is not None

    def test_model_is_sklearn_pipeline(self, pipeline):
        """Loaded object must be a scikit-learn Pipeline."""
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline), \
            f"Expected sklearn Pipeline, got {type(pipeline)}"

    def test_pipeline_has_steps(self, pipeline):
        """Pipeline must have at least one step."""
        assert len(pipeline.steps) >= 1, "Pipeline has no steps!"


# ══════════════════════════════════════════════════════════════════
#  2. DATASET TESTS
# ══════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    "Age", "Annual_Income", "Website_Visits", "Previous_Purchases",
    "Membership_Type", "Preferred_Category", "City_Tier",
    "Discount_Used", "Amount_Spent",
]

class TestDataset:
    def test_dataset_loads(self, dataset):
        """Dataset CSV must load successfully."""
        assert isinstance(dataset, pd.DataFrame)

    def test_dataset_not_empty(self, dataset):
        """Dataset must have records."""
        assert len(dataset) > 0, "Dataset is empty!"

    def test_required_columns_present(self, dataset):
        """All required feature + target columns must be present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in dataset.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_nulls_in_features(self, dataset):
        """Feature columns should have no null values after cleaning."""
        feature_cols = [c for c in REQUIRED_COLUMNS if c != "Amount_Spent"]
        null_counts   = dataset[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        assert not cols_with_nulls, f"Null values found: {cols_with_nulls}"

    def test_target_column_positive(self, dataset):
        """Amount_Spent must be strictly positive (spending can't be negative)."""
        assert (dataset["Amount_Spent"] > 0).all(), \
            "Found non-positive values in Amount_Spent!"

    def test_age_reasonable(self, dataset):
        """Age should be within [1, 120]."""
        assert dataset["Age"].between(1, 120).all(), \
            f"Age out of range: min={dataset['Age'].min()}, max={dataset['Age'].max()}"

    def test_membership_types_valid(self, dataset):
        """Membership_Type must be one of Bronze, Silver, Gold."""
        valid = {"Bronze", "Silver", "Gold"}
        actual = set(dataset["Membership_Type"].unique())
        assert actual.issubset(valid), \
            f"Unexpected Membership_Type values: {actual - valid}"

    def test_discount_used_values(self, dataset):
        """Discount_Used must only contain Yes/No."""
        valid   = {"Yes", "No"}
        actual  = set(dataset["Discount_Used"].unique())
        assert actual.issubset(valid), \
            f"Unexpected Discount_Used values: {actual - valid}"

    def test_dataset_minimum_size(self, dataset):
        """Dataset should have at least 100 records for meaningful ML."""
        assert len(dataset) >= 100, \
            f"Dataset too small: only {len(dataset)} records"

    def test_income_range(self, dataset):
        """Annual Income should be positive."""
        assert (dataset["Annual_Income"] > 0).all(), \
            "Found non-positive Annual_Income values!"


# ══════════════════════════════════════════════════════════════════
#  3. PREDICTION / INFERENCE TESTS
# ══════════════════════════════════════════════════════════════════

class TestPrediction:
    def test_predict_returns_value(self, pipeline, sample_customer):
        """Pipeline.predict() must return a result."""
        result = pipeline.predict(sample_customer)
        assert result is not None

    def test_predict_returns_float(self, pipeline, sample_customer):
        """Prediction must be a numeric type."""
        result = pipeline.predict(sample_customer)[0]
        assert isinstance(result, (float, int, np.floating, np.integer)), \
            f"Expected numeric prediction, got {type(result)}"

    def test_predict_positive_value(self, pipeline, sample_customer):
        """Spending prediction must be a positive number."""
        result = pipeline.predict(sample_customer)[0]
        assert result > 0, f"Predicted negative spending: {result}"

    def test_predict_realistic_range(self, pipeline, sample_customer, dataset):
        """Prediction should be within a reasonable range relative to training data."""
        result   = pipeline.predict(sample_customer)[0]
        data_min = dataset["Amount_Spent"].min() * 0.1  # allow 10× underpredict
        data_max = dataset["Amount_Spent"].max() * 10   # allow 10× overpredict
        assert data_min <= result <= data_max, \
            f"Prediction {result:.2f} outside plausible range [{data_min:.2f}, {data_max:.2f}]"

    def test_predict_single_row(self, pipeline, sample_customer):
        """Pipeline should handle a single-row DataFrame."""
        result = pipeline.predict(sample_customer)
        assert len(result) == 1

    def test_predict_multiple_customers(self, pipeline):
        """Pipeline should handle batch predictions."""
        batch = pd.DataFrame({
            "Age":                [25, 45, 60],
            "Annual_Income":      [300_000, 700_000, 1_500_000],
            "Website_Visits":     [5, 15, 2],
            "Previous_Purchases": [2, 10, 20],
            "Membership_Type":    ["Bronze", "Silver", "Gold"],
            "Preferred_Category": ["Fashion", "Home", "Electronics"],
            "City_Tier":          ["Tier 3", "Tier 2", "Tier 1"],
            "Discount_Used":      ["No", "Yes", "No"],
        })
        results = pipeline.predict(batch)
        assert len(results) == 3, "Expected 3 predictions for batch of 3"

    def test_predict_deterministic(self, pipeline, sample_customer):
        """Same input must always produce same output (no randomness at inference)."""
        r1 = pipeline.predict(sample_customer)[0]
        r2 = pipeline.predict(sample_customer)[0]
        assert r1 == r2, f"Non-deterministic prediction: {r1} vs {r2}"

    def test_predict_silver_member(self, pipeline):
        """Silver membership prediction should work without error."""
        customer = pd.DataFrame({
            "Age":                [40],
            "Annual_Income":      [500_000],
            "Website_Visits":     [8],
            "Previous_Purchases": [6],
            "Membership_Type":    ["Silver"],
            "Preferred_Category": ["Beauty"],
            "City_Tier":          ["Tier 2"],
            "Discount_Used":      ["No"],
        })
        result = pipeline.predict(customer)[0]
        assert result > 0

    def test_predict_bronze_member(self, pipeline):
        """Bronze membership prediction should work without error."""
        customer = pd.DataFrame({
            "Age":                [22],
            "Annual_Income":      [200_000],
            "Website_Visits":     [3],
            "Previous_Purchases": [1],
            "Membership_Type":    ["Bronze"],
            "Preferred_Category": ["Fashion"],
            "City_Tier":          ["Tier 3"],
            "Discount_Used":      ["Yes"],
        })
        result = pipeline.predict(customer)[0]
        assert result > 0

    def test_predict_high_income_customer(self, pipeline):
        """High-income Gold customer should predict higher than low-income Bronze."""
        high = pd.DataFrame({
            "Age":                [45],
            "Annual_Income":      [5_000_000],
            "Website_Visits":     [30],
            "Previous_Purchases": [50],
            "Membership_Type":    ["Gold"],
            "Preferred_Category": ["Electronics"],
            "City_Tier":          ["Tier 1"],
            "Discount_Used":      ["No"],
        })
        low = pd.DataFrame({
            "Age":                [22],
            "Annual_Income":      [150_000],
            "Website_Visits":     [1],
            "Previous_Purchases": [1],
            "Membership_Type":    ["Bronze"],
            "Preferred_Category": ["Fashion"],
            "City_Tier":          ["Tier 3"],
            "Discount_Used":      ["No"],
        })
        high_pred = pipeline.predict(high)[0]
        low_pred  = pipeline.predict(low)[0]
        assert high_pred >= low_pred, \
            f"Expected high income ({high_pred:.0f}) >= low income ({low_pred:.0f})"


# ══════════════════════════════════════════════════════════════════
#  4. EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_predict_all_city_tiers(self, pipeline):
        """All three City Tiers must work."""
        for tier in ["Tier 1", "Tier 2", "Tier 3"]:
            customer = pd.DataFrame({
                "Age":                [35],
                "Annual_Income":      [600_000],
                "Website_Visits":     [10],
                "Previous_Purchases": [5],
                "Membership_Type":    ["Silver"],
                "Preferred_Category": ["Home"],
                "City_Tier":          [tier],
                "Discount_Used":      ["No"],
            })
            result = pipeline.predict(customer)[0]
            assert result > 0, f"Negative prediction for {tier}"

    def test_predict_all_categories(self, pipeline, dataset):
        """All preferred categories in dataset must work."""
        for cat in dataset["Preferred_Category"].unique():
            customer = pd.DataFrame({
                "Age":                [30],
                "Annual_Income":      [500_000],
                "Website_Visits":     [8],
                "Previous_Purchases": [5],
                "Membership_Type":    ["Silver"],
                "Preferred_Category": [cat],
                "City_Tier":          ["Tier 2"],
                "Discount_Used":      ["No"],
            })
            result = pipeline.predict(customer)[0]
            assert result > 0, f"Invalid prediction for category: {cat}"

    def test_predict_discount_both_values(self, pipeline):
        """Both Yes/No discount values must work."""
        for disc in ["Yes", "No"]:
            customer = pd.DataFrame({
                "Age":                [28],
                "Annual_Income":      [400_000],
                "Website_Visits":     [7],
                "Previous_Purchases": [4],
                "Membership_Type":    ["Bronze"],
                "Preferred_Category": ["Beauty"],
                "City_Tier":          ["Tier 1"],
                "Discount_Used":      [disc],
            })
            result = pipeline.predict(customer)[0]
            assert result > 0, f"Invalid prediction for Discount_Used={disc}"

    def test_task_example_customer(self, pipeline):
        """Test the exact example from the task specification."""
        example = pd.DataFrame({
            "Age":                [32],
            "Annual_Income":      [800_000],
            "Website_Visits":     [12],
            "Previous_Purchases": [8],
            "Membership_Type":    ["Gold"],
            "Preferred_Category": ["Electronics"],
            "City_Tier":          ["Tier 1"],
            "Discount_Used":      ["Yes"],
        })
        result = pipeline.predict(example)[0]
        print(f"\n📦 Task Example Prediction: ₹{result:,.2f}")
        assert result > 0, "Task example prediction must be positive!"
