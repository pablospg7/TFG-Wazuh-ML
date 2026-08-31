import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# 1. CARGAR DATASETS
# =========================
dataset = pd.read_csv("dataset_final.csv", low_memory=False)
alerts = pd.read_csv("wazuh_alerts_all.csv", low_memory=False)

print("=== DATASET FINAL ===")
print("Filas:", len(dataset))
print(dataset["label"].value_counts())

print("\n=== WAZUH ALERTS ===")
print("Filas:", len(alerts))
if "label" in alerts.columns:
    print(alerts["label"].value_counts())

# =========================
# 2. FEATURES LIMPIAS
# =========================
safe_text_cols = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser"
]

usable_cols = [c for c in safe_text_cols if c in dataset.columns]

print("\n=== COLUMNAS USADAS POR EL ML ===")
for c in usable_cols:
    print("-", c)

dataset["text"] = ""
for c in usable_cols:
    dataset["text"] += dataset[c].fillna("").astype(str) + " "

X = dataset["text"]
y = dataset["label"]

# =========================
# 3. SPLIT Y ENTRENAMIENTO
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)
ml_pred = model.predict(X_test_vec)

# Guardar predicciones ML
ml_results = pd.DataFrame({
    "text": X_test.values,
    "real": y_test.values,
    "ml_pred": ml_pred
})

# =========================
# 4. PREPARAR ALERTS DE WAZUH
# =========================
if "_source.full_log" in alerts.columns:
    alerts["full_log_str"] = alerts["_source.full_log"].fillna("").astype(str).str.strip()
else:
    alerts["full_log_str"] = ""

alert_logs = set(alerts["full_log_str"])

# En el texto del test buscamos si Wazuh alertó algo equivalente
def wazuh_detected(text):
    for log in alert_logs:
        if log and log in text:
            return 1
    return 0

ml_results["wazuh_pred"] = ml_results["text"].fillna("").astype(str).apply(wazuh_detected)

# =========================
# 5. MÉTRICAS DEL ML
# =========================
ml_acc = accuracy_score(ml_results["real"], ml_results["ml_pred"])
ml_prec = precision_score(ml_results["real"], ml_results["ml_pred"])
ml_rec = recall_score(ml_results["real"], ml_results["ml_pred"])
ml_f1 = f1_score(ml_results["real"], ml_results["ml_pred"])

print("\n=== RANDOM FOREST (ML) ===")
print("Accuracy :", round(ml_acc, 4))
print("Precision:", round(ml_prec, 4))
print("Recall   :", round(ml_rec, 4))
print("F1-score :", round(ml_f1, 4))

# =========================
# 6. MÉTRICAS DE WAZUH
# =========================
wazuh_acc = accuracy_score(ml_results["real"], ml_results["wazuh_pred"])
wazuh_prec = precision_score(ml_results["real"], ml_results["wazuh_pred"], zero_division=0)
wazuh_rec = recall_score(ml_results["real"], ml_results["wazuh_pred"], zero_division=0)
wazuh_f1 = f1_score(ml_results["real"], ml_results["wazuh_pred"], zero_division=0)

print("\n=== WAZUH ===")
print("Accuracy :", round(wazuh_acc, 4))
print("Precision:", round(wazuh_prec, 4))
print("Recall   :", round(wazuh_rec, 4))
print("F1-score :", round(wazuh_f1, 4))

# =========================
# 7. MATRIZ DE CONFUSIÓN
# =========================
cm_ml = confusion_matrix(ml_results["real"], ml_results["ml_pred"])
disp_ml = ConfusionMatrixDisplay(confusion_matrix=cm_ml)
disp_ml.plot()
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_random_forest_final.png", dpi=200)
plt.close()

cm_wazuh = confusion_matrix(ml_results["real"], ml_results["wazuh_pred"])
disp_w = ConfusionMatrixDisplay(confusion_matrix=cm_wazuh)
disp_w.plot()
plt.title("Confusion Matrix - Wazuh")
plt.tight_layout()
plt.savefig("confusion_matrix_wazuh.png", dpi=200)
plt.close()

# =========================
# 8. COMPARACIÓN FINAL
# =========================
comparison = pd.DataFrame([
    {
        "system": "Random Forest",
        "accuracy": ml_acc,
        "precision": ml_prec,
        "recall": ml_rec,
        "f1_score": ml_f1
    },
    {
        "system": "Wazuh",
        "accuracy": wazuh_acc,
        "precision": wazuh_prec,
        "recall": wazuh_rec,
        "f1_score": wazuh_f1
    }
])

comparison.to_csv("wazuh_vs_ml_comparison.csv", index=False)

print("\n=== COMPARACIÓN FINAL ===")
print(comparison)

# =========================
# 9. CASOS INTERESANTES
# =========================
only_ml = ml_results[(ml_results["ml_pred"] == 1) & (ml_results["wazuh_pred"] == 0)]
only_wazuh = ml_results[(ml_results["ml_pred"] == 0) & (ml_results["wazuh_pred"] == 1)]
both_detect = ml_results[(ml_results["ml_pred"] == 1) & (ml_results["wazuh_pred"] == 1)]

only_ml.to_csv("detected_only_by_ml.csv", index=False)
only_wazuh.to_csv("detected_only_by_wazuh.csv", index=False)
both_detect.to_csv("detected_by_both.csv", index=False)
ml_results.to_csv("ml_test_predictions_final.csv", index=False)

print("\nArchivos generados:")
print("- wazuh_vs_ml_comparison.csv")
print("- ml_test_predictions_final.csv")
print("- detected_only_by_ml.csv")
print("- detected_only_by_wazuh.csv")
print("- detected_by_both.csv")
print("- confusion_matrix_random_forest_final.png")
print("- confusion_matrix_wazuh.png")