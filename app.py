import joblib, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import compute_metrics

st.set_page_config(page_title="Classifier Playground", layout="wide")
st.title("🔍 Classification Model Playground")
st.write("Upload **test data** (features + target column), pick a model, see how it performs.")

MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
          "KNN": "knn", "Naive Bayes": "naive_bayes", "SVM": "svm"}

# --- (a) CSV upload ---
file = st.sidebar.file_uploader("Upload test CSV", type="csv")
# --- (b) Model dropdown ---
choice = st.sidebar.selectbox("Choose a model", list(MODELS))

if file is None:
    st.info("👈 Upload test_data.csv from the repo to begin.")
    st.stop()

df = pd.read_csv(file)
target = st.sidebar.selectbox("Target column", df.columns, index=len(df.columns) - 1)
X, y = df.drop(columns=[target]), df[target]
st.subheader("Data preview"); st.dataframe(df.head())

# Load the SAVED model - no retraining happens here
model = joblib.load(f"model/{MODELS[choice]}.joblib")

# --- (c) Metrics ---
m = compute_metrics(model, X, y)
st.subheader(f"Metrics: {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.3f}")

# --- (d) Confusion matrix + report ---
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, model.predict(X)), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    st.text(classification_report(y, model.predict(X)))

# Bonus: all models side by side
if st.checkbox("Compare all 5 models on this data"):
    st.dataframe(pd.DataFrame({n: compute_metrics(joblib.load(f"model/{f}.joblib"), X, y)
                               for n, f in MODELS.items()}).T.round(4))
