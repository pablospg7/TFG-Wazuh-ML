import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
INPUT_DATASET = "dataset_final.csv"
OUTPUT_ALL = "ml_readable_view_clean.csv"
OUTPUT_ATTACKS = "ml_only_attacks_readable.csv"
OUTPUT_TRUE_ATTACKS = "ml_true_attacks_readable_clean.csv"

# =========================
# 1. CARGAR DATASET
# =========================
df = pd.read_csv(INPUT_DATASET, low_memory=False)

cols = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser"
]

df["text"] = ""
for c in cols:
    if c in df.columns:
        df["text"] += df[c].fillna("").astype(str) + " "

X = df["text"]
y = df["label"]

# =========================
# 2. SPLIT IGUAL QUE ANTES
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 4. RANDOM FOREST CLEAN
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)

# =========================
# 5. CLASIFICACIÓN HUMANA SIMPLE
# =========================
def estimate_attack_type(text):
    t = str(text).lower()

    ssh_keywords = [
        "sshd", "failed password", "authentication failed",
        "preauth", "pam", "maximum authentication attempts exceeded",
        "invalid user", "ssh"
    ]

    web_keywords = [
        "gobuster", "dirb", "get ", "404", "http",
        "apache", "forbidden", "suspicious url"
    ]

    ssh_hits = sum(1 for k in ssh_keywords if k in t)
    web_hits = sum(1 for k in web_keywords if k in t)

    if ssh_hits > web_hits and ssh_hits > 0:
        return "ssh_attack"
    if web_hits > ssh_hits and web_hits > 0:
        return "web_attack"
    if ssh_hits > 0 and web_hits > 0:
        return "mixed_or_unclear"
    return "unknown"

# =========================
# 6. GUARDAR VISTA LEGIBLE
# =========================
results = pd.DataFrame({
    "log": X_test.values,
    "etiqueta_real": y_test.values,
    "prediccion_ml": pred
})

results["tipo_ataque_estimado"] = results["log"].apply(estimate_attack_type)
results["prediccion_ml_texto"] = results["prediccion_ml"].map({0: "normal", 1: "ataque"})
results["etiqueta_real_texto"] = results["etiqueta_real"].map({0: "normal", 1: "ataque"})
results["detectado_correctamente"] = results["etiqueta_real"] == results["prediccion_ml"]

# todos los eventos del test
results.to_csv(OUTPUT_ALL, index=False)

# solo lo que ML marca como ataque
ml_attacks = results[results["prediccion_ml"] == 1].copy()
ml_attacks.to_csv(OUTPUT_ATTACKS, index=False)

# solo ataques reales correctamente detectados
ml_true_attacks = results[
    (results["prediccion_ml"] == 1) & (results["etiqueta_real"] == 1)
].copy()
ml_true_attacks.to_csv(OUTPUT_TRUE_ATTACKS, index=False)

print("Archivos generados:")
print("-", OUTPUT_ALL)
print("-", OUTPUT_ATTACKS)
print("-", OUTPUT_TRUE_ATTACKS)

print("\nResumen:")
print("Total eventos test:", len(results))
print("Eventos marcados como ataque por ML:", len(ml_attacks))
print("Ataques reales correctamente detectados:", len(ml_true_attacks))
print("Falsos positivos:", len(results[(results['etiqueta_real'] == 0) & (results['prediccion_ml'] == 1)]))
print("Falsos negativos:", len(results[(results['etiqueta_real'] == 1) & (results['prediccion_ml'] == 0)]))

print("\nTipos estimados entre ataques detectados:")
print(ml_attacks["tipo_ataque_estimado"].value_counts())