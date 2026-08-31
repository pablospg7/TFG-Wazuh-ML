import re
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# 1. CARGAR DATASET
# =====================================================
df = pd.read_csv("dataset_final.csv", low_memory=False)

cols = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser"
]

df["text"] = ""

for c in cols:
    if c in df.columns:
        df["text"] += df[c].fillna("").astype(str) + " "

# =====================================================
# 2. NLP: LIMPIEZA Y NORMALIZACIÓN DEL TEXTO
# =====================================================
custom_stopwords = {
    "ubuntu",
    "victima",
    "vboxuser",
    "mozilla",
    "windows",
    "compatible",
    "msie",
    "nt",
    "http",
    "https"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"192\s+168\s+\d+\s+\d+", " ", text)
    text = re.sub(r"[_:/\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =====================================================
# 3. TRAIN / TEST
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 4. TF-IDF: TEXTO -> VECTORES NUMÉRICOS
# =====================================================
vectorizer = TfidfVectorizer(
    max_features=5000,
    token_pattern=r"(?u)\b[a-zA-Z_]{3,}\b",
    stop_words=list(custom_stopwords)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()

# =====================================================
# 5. RANDOM FOREST
# =====================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_vec, y_train)

# =====================================================
# 6. SHAP ESTABLE CON BACKGROUND
# =====================================================
X_background = X_train_vec[:200].toarray().astype("float64")
X_sample = X_test_vec.toarray().astype("float64")

explainer = shap.TreeExplainer(
    model,
    data=X_background,
    feature_perturbation="interventional",
    model_output="probability"
)

shap_values = explainer.shap_values(
    X_sample,
    check_additivity=False
)

# Clase 1 = ataque
if isinstance(shap_values, list):
    shap_attack = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_attack = shap_values[:, :, 1]
else:
    shap_attack = shap_values

# =====================================================
# 7. SHAP BAR
# =====================================================
plt.figure()

shap.summary_plot(
    shap_attack,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.savefig(
    "shap_bar_rf_clean.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 8. SHAP BEESWARM
# =====================================================
plt.figure()

shap.summary_plot(
    shap_attack,
    X_sample,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.savefig(
    "shap_beeswarm_rf_clean.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 9. TOP SHAP FEATURES CSV
# =====================================================
mean_abs_shap = np.abs(shap_attack).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:30]

top_shap = pd.DataFrame({
    "feature": feature_names[top_idx],
    "mean_abs_shap": mean_abs_shap[top_idx]
})

top_shap.to_csv(
    "top_shap_features_rf_clean.csv",
    index=False
)

print("SHAP terminado.")
print("Archivos generados:")
print("- shap_bar_rf_clean.png")
print("- shap_beeswarm_rf_clean.png")
print("- top_shap_features_rf_clean.csv")