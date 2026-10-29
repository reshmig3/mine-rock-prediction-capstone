import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv("data/sonar.csv", header=None)
try:
    float(df.iloc[0, 0])
except ValueError:
    df = pd.read_csv("data/sonar.csv")  # file has a header row

X = df.iloc[:, :60].astype(float)
y = df.iloc[:, 60].astype(str).str.strip()
print("Rows:", len(df), "| Classes:", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM (RBF)": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
    print(f"{name}: CV accuracy {scores.mean():.3f} +/- {scores.std():.3f}")

best = max(cv_results, key=lambda n: cv_results[n]["mean"])
print("Best by cross-validation:", best)

final = Pipeline([("scaler", StandardScaler()), ("model", models[best])])
final.fit(X_train, y_train)
pred = final.predict(X_test)

metrics = {
    "model": best,
    "train_rows": len(X_train), "test_rows": len(X_test),
    "test_accuracy": round(accuracy_score(y_test, pred), 4),
    "precision_mine": round(precision_score(y_test, pred, pos_label="M"), 4),
    "recall_mine": round(recall_score(y_test, pred, pos_label="M"), 4),
    "f1_mine": round(f1_score(y_test, pred, pos_label="M"), 4),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=["M", "R"]).tolist(),
    "labels": ["M", "R"],
    "cross_validation": cv_results,
}
print(json.dumps(metrics, indent=2))

joblib.dump(final, "model/mine_rock_pipeline.joblib")
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, labels=["M", "R"], display_labels=["Mine", "Rock"])
plt.title("Confusion Matrix (held-out test set)")
plt.savefig("model/confusion_matrix.png", dpi=150, bbox_inches="tight")