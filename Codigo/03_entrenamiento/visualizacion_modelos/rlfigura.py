import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Cargar dataset
# ============================================================

df = pd.read_csv("dataset_final.csv", low_memory=False)

safe_text_cols = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser"
]

usable_cols = [c for c in safe_text_cols if c in df.columns]

df["text"] = ""
for c in usable_cols:
    df["text"] += df[c].fillna("").astype(str) + " "

X = df["text"]
y = df["label"]


# ============================================================
# 2. División train/test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. TF-IDF + Regresión Logística
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    token_pattern=r"(?u)\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{3,}\b"
)

X_train_vec = vectorizer.fit_transform(X_train)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_vec, y_train)

feature_names = vectorizer.get_feature_names_out()
coefficients = lr_model.coef_[0]


# ============================================================
# 4. Seleccionar términos representativos
# ============================================================

preferred_terms = [
    "sshd",
    "failed",
    "password",
    "gobuster",
    "get",
    "http",
    "port",
    "authentication",
    "invalid",
    "apache"
]

available_terms = set(feature_names)
selected_terms = [t for t in preferred_terms if t in available_terms]

# Completar con términos con mayor peso absoluto en la Regresión Logística
top_idx = abs(coefficients).argsort()[-80:][::-1]

for i in top_idx:
    term = feature_names[i]
    if term not in selected_terms and not term.isdigit():
        selected_terms.append(term)
    if len(selected_terms) >= 6:
        break

top_terms = selected_terms[:6]

if len(top_terms) < 4:
    raise ValueError("No hay suficientes términos para generar la figura.")


# ============================================================
# 5. Funciones auxiliares de dibujo
# ============================================================

def add_box(ax, x, y, w, h, text, fontsize=10,
            facecolor="#FFFFFF", edgecolor="#222222"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.7,
        facecolor=facecolor,
        edgecolor=edgecolor
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=17,
        linewidth=1.6,
        color="#222222"
    )
    ax.add_patch(arrow)


# ============================================================
# 6. Crear figura conceptual
# ============================================================

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.suptitle(
    "Funcionamiento conceptual de la Regresión Logística aplicada a alertas",
    fontsize=18,
    fontweight="bold",
    y=0.97
)

ax.text(
    0.5,
    0.91,
    "El modelo combina los pesos TF-IDF de los términos y estima la probabilidad de pertenecer a una clase.",
    ha="center",
    va="center",
    fontsize=11
)

# Entrada
add_box(
    ax,
    0.04,
    0.56,
    0.18,
    0.20,
    "Alertas de Wazuh\n\nCampos usados:\nfull_log\nurl\ndstuser",
    fontsize=10,
    facecolor="#EAF2FF",
    edgecolor="#2E5AAC"
)

# TF-IDF
terms_text = (
    "Representación TF-IDF\n\n"
    "Términos representativos:\n"
    + "\n".join(top_terms[:6])
)

add_box(
    ax,
    0.29,
    0.56,
    0.22,
    0.20,
    terms_text,
    fontsize=9,
    facecolor="#FFF6DF",
    edgecolor="#B7791F"
)

add_arrow(ax, 0.22, 0.66, 0.29, 0.66)

# Combinación ponderada
formula_text = (
    "Combinación ponderada\n\n"
    f"w₁·TF-IDF('{top_terms[0]}')\n"
    f"+ w₂·TF-IDF('{top_terms[1]}')\n"
    f"+ w₃·TF-IDF('{top_terms[2]}')\n"
    "+ ..."
)

add_box(
    ax,
    0.58,
    0.56,
    0.24,
    0.20,
    formula_text,
    fontsize=9,
    facecolor="#EAFBF0",
    edgecolor="#2F855A"
)

add_arrow(ax, 0.51, 0.66, 0.58, 0.66)

# Probabilidad
add_box(
    ax,
    0.58,
    0.28,
    0.24,
    0.16,
    "Función logística\n\nProbabilidad de ataque\nP(ataque)",
    fontsize=10,
    facecolor="#F3E8FF",
    edgecolor="#6B46C1"
)

add_arrow(ax, 0.70, 0.56, 0.70, 0.44)

# Salida
add_box(
    ax,
    0.87,
    0.42,
    0.11,
    0.16,
    "Decisión final\n\nNormal\n/\nAtaque",
    fontsize=10,
    facecolor="#FFECEC",
    edgecolor="#C53030"
)

add_arrow(ax, 0.82, 0.36, 0.87, 0.50)

# Nota inferior
ax.text(
    0.5,
    0.08,
    "Figura conceptual generada a partir del conjunto de datos del proyecto. "
    "No representa todos los coeficientes del modelo, sino una simplificación del flujo general de decisión.",
    ha="center",
    va="center",
    fontsize=10
)

plt.tight_layout()
plt.savefig("regresion_logistica_conceptual_tfg.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figura guardada como regresion_logistica_conceptual_tfg.png")
print("Términos usados en la figura:", top_terms)