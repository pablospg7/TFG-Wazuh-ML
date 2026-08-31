import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

df = pd.read_csv("dataset_final.csv", low_memory=False)

print("=== DATASET ===")
print("Filas:", len(df))
print(df["label"].value_counts())

# Versión más limpia: menos riesgo de sobreajuste
safe_text_cols = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser"
]

usable_cols = [c for c in safe_text_cols if c in df.columns]

print("\n=== COLUMNAS USADAS ===")
for c in usable_cols:
    print("-", c)

df["text"] = ""
for c in usable_cols:
    df["text"] += df[c].fillna("").astype(str) + " "

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

def eval_model(model, name):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))

    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()

    print("False Positives:", int(fp))
    print("False Negatives:", int(fn))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_clean_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }

results = []
results.append(eval_model(LogisticRegression(max_iter=1000), "Logistic Regression Clean"))
results.append(eval_model(RandomForestClassifier(n_estimators=200, random_state=42), "Random Forest Clean"))

comparison = pd.DataFrame(results)
comparison.to_csv("model_comparison_clean.csv", index=False)

print("\n=== COMPARACIÓN CLEAN ===")
print(comparison)