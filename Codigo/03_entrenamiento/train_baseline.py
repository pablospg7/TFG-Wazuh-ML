import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# 1. CARGAR DATASET
# =========================
df = pd.read_csv("dataset_final.csv", low_memory=False)

print("=== INFO DATASET ===")
print("Filas:", len(df))
print("Columnas:", len(df.columns))
print("\nDistribución de labels:")
print(df["label"].value_counts())

# =========================
# 2. DETECTAR POSIBLE FUGA DE INFORMACIÓN
# =========================
leakage_cols = [
    "_source.rule.level",
    "_source.rule.description",
    "_source.rule.groups",
    "_source.rule.id",
    "_index",
    "_id",
    "_version",
    "_score",
    "_source.id"
]

present_leakage = [c for c in leakage_cols if c in df.columns]

print("\n=== COLUMNAS CON POSIBLE DATA LEAKAGE ===")
if present_leakage:
    for col in present_leakage:
        print("-", col)
else:
    print("No se detectaron columnas peligrosas.")

# =========================
# 3. COLUMNAS SEGURAS PARA FEATURES
# =========================
safe_text_cols = [
    "_source.full_log",
    "_source.location",
    "_source.decoder.name",
    "_source.agent.name",
    "_source.agent.ip",
    "_source.data.srcip",
    "_source.data.dstuser",
    "_source.predecoder.program_name",
    "_source.predecoder.hostname",
    "_source.data.url"
]

usable_cols = [c for c in safe_text_cols if c in df.columns]

print("\n=== COLUMNAS USADAS COMO TEXTO ===")
for col in usable_cols:
    print("-", col)

# rellenar nulos y construir texto
df["text"] = ""
for col in usable_cols:
    df["text"] += df[col].fillna("").astype(str) + " "

# target
X = df["text"]
y = df["label"]

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== SPLIT ===")
print("Train:", len(X_train))
print("Test:", len(X_test))

# =========================
# 5. TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 6. FUNCIÓN DE EVALUACIÓN
# =========================
def evaluate_model(model, model_name, X_train_vec, X_test_vec, y_train, y_test, X_test_raw):
    print(f"\n===== {model_name} =====")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # guardar métricas en csv resumen
    metrics_df = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positives": int(((y_test == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y_test == 1) & (y_pred == 0)).sum())
    }])
    metrics_df.to_csv(f"metrics_{model_name.lower().replace(' ', '_')}.csv", index=False)

    # falsos positivos y negativos
    results = pd.DataFrame({
        "text": X_test_raw.values,
        "real": y_test.values,
        "pred": y_pred
    })

    false_positives = results[(results["real"] == 0) & (results["pred"] == 1)]
    false_negatives = results[(results["real"] == 1) & (results["pred"] == 0)]

    false_positives.to_csv(
        f"false_positives_{model_name.lower().replace(' ', '_')}.csv",
        index=False
    )
    false_negatives.to_csv(
        f"false_negatives_{model_name.lower().replace(' ', '_')}.csv",
        index=False
    )

    print("False Positives:", len(false_positives))
    print("False Negatives:", len(false_negatives))

    # matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives)
    }

# =========================
# 7. ENTRENAR MODELOS
# =========================
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

results_lr = evaluate_model(
    lr_model,
    "Logistic Regression",
    X_train_vec,
    X_test_vec,
    y_train,
    y_test,
    X_test
)

results_rf = evaluate_model(
    rf_model,
    "Random Forest",
    X_train_vec,
    X_test_vec,
    y_train,
    y_test,
    X_test
)

# =========================
# 8. COMPARATIVA FINAL
# =========================
comparison = pd.DataFrame([results_lr, results_rf])
comparison.to_csv("model_comparison.csv", index=False)

print("\n=== COMPARATIVA FINAL ===")
print(comparison)

# escoger mejor modelo por F1
best_model = comparison.sort_values("f1_score", ascending=False).iloc[0]
print("\n=== MEJOR MODELO SEGÚN F1 ===")
print(best_model)