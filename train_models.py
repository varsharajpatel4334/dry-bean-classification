"""Train five classification models and save their evaluation results."""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils import compute_metrics


# Dry Bean dataset configuration
DATA_PATH = "dry_bean.csv"
TARGET_COL = "Class"


# Load the dataset
df = pd.read_csv(DATA_PATH)

# Remove duplicate records, if any
df = df.drop_duplicates().reset_index(drop=True)

# Separate input features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("Dataset shape:", df.shape)
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", sorted(y.unique()))
print("Missing values:", df.isnull().sum().sum())


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Define all five required classification models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    )
}


# Create model folder if it does not exist
os.makedirs("model", exist_ok=True)

results = {}

# Train, evaluate and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    results[name] = compute_metrics(
        model,
        X_test,
        y_test
    )

    model_filename = name.replace(" ", "_").lower() + ".joblib"
    model_path = os.path.join("model", model_filename)

    joblib.dump(model, model_path)

    print(f"Saved: {model_path}")


# Make the comparison table
results_df = pd.DataFrame(results).T.round(4)

print("\nModel comparison:")
print(results_df)


# Save model evaluation results
results_df.to_csv("metrics_table.csv")


# Save unseen testing data for the Streamlit application
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv("test_data.csv", index=False)

print("\nCreated successfully:")
print("1. Five saved model files inside the model folder")
print("2. metrics_table.csv")
print("3. test_data.csv")