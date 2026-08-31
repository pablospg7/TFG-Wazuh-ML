from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


TEXT_COLUMNS = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser",
]

# Términos que resultan útiles para explicar los escenarios del laboratorio.
SECURITY_TERMS = {
    "authentication", "authenticating", "attempts", "disconnecting",
    "error", "exceeded", "failed", "failure", "failures", "get",
    "gobuster", "http", "maximum", "password", "port", "retries",
    "root", "session", "ssh", "ssh2", "sshd", "systemd",
}

# Términos que pueden ser correctos, pero aportan poco a la figura.
VISUAL_NOISE = {
    "ubuntu", "victima", "vboxuser",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "msie", "nt", "compatible", "windows", "amd64",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dos figuras para documentar Random Forest: "
            "la distribución de la profundidad de los 200 árboles y "
            "un árbol ilustrativo con términos legibles."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="dataset_final.csv",
        help="Ruta al CSV. Por defecto: dataset_final.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="salida_random_forest_texto_realmente_grande",
        help="Carpeta de salida. Por defecto: salida_random_forest_texto_realmente_grande",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> None:
    required = [*TEXT_COLUMNS, "label"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            + ", ".join(missing)
        )

    labels = set(df["label"].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            "La columna 'label' debe contener únicamente 0 y 1."
        )


def build_text(df: pd.DataFrame) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="object")

    for column in TEXT_COLUMNS:
        text = text + df[column].fillna("").astype(str) + " "

    return text.str.strip()


def first_levels_terms(
    estimator,
    feature_names: np.ndarray,
    max_depth: int = 2,
) -> list[str]:
    tree = estimator.tree_
    pending = [(0, 0)]
    terms: list[str] = []

    while pending:
        node, depth = pending.pop(0)

        if node < 0 or depth > max_depth:
            continue

        feature_index = tree.feature[node]

        if feature_index >= 0:
            terms.append(str(feature_names[feature_index]))
            pending.append((tree.children_left[node], depth + 1))
            pending.append((tree.children_right[node], depth + 1))

    return terms


def choose_readable_tree(
    model: RandomForestClassifier,
    feature_names: np.ndarray,
    summary: pd.DataFrame,
) -> tuple[int, list[str]]:
    """
    Selecciona un árbol solo para ilustrar el funcionamiento del modelo.

    Se favorecen árboles cuyos primeros niveles:
    - no estén dominados por números;
    - contengan términos relacionados con los escenarios;
    - mantengan una estructura razonable respecto al conjunto del bosque.

    La selección no modifica el entrenamiento ni las predicciones.
    """
    depth_median = summary["profundidad"].median()
    leaves_median = summary["hojas"].median()

    depth_std = summary["profundidad"].std() or 1.0
    leaves_std = summary["hojas"].std() or 1.0

    candidates = []

    for tree_index, estimator in enumerate(model.estimators_):
        terms = first_levels_terms(
            estimator,
            feature_names,
            max_depth=2,
        )

        numeric_terms = sum(term.isdigit() for term in terms)
        noise_terms = sum(term.lower() in VISUAL_NOISE for term in terms)
        security_terms = sum(term.lower() in SECURITY_TERMS for term in terms)
        alphabetic_terms = sum(
            bool(re.fullmatch(r"[A-Za-z][A-Za-z_-]*", term))
            for term in terms
        )

        structural_distance = (
            abs(estimator.tree_.max_depth - depth_median) / depth_std
            + abs(estimator.tree_.n_leaves - leaves_median) / leaves_std
        )

        score = (
            security_terms * 8
            + alphabetic_terms
            - numeric_terms * 12
            - noise_terms * 5
            - structural_distance
        )

        candidates.append({
            "arbol": tree_index,
            "puntuacion_visual": score,
            "terminos_primeros_niveles": ", ".join(terms),
            "numero_terminos_numericos": numeric_terms,
            "numero_terminos_seguridad": security_terms,
        })

    candidates_df = pd.DataFrame(candidates)

    # Se priorizan candidatos sin términos puramente numéricos.
    readable = candidates_df[
        candidates_df["numero_terminos_numericos"] == 0
    ]

    if readable.empty:
        readable = candidates_df

    selected = readable.sort_values(
        ["puntuacion_visual", "numero_terminos_seguridad"],
        ascending=[False, False],
    ).iloc[0]

    selected_index = int(selected["arbol"])
    selected_terms = first_levels_terms(
        model.estimators_[selected_index],
        feature_names,
        max_depth=2,
    )

    return selected_index, selected_terms


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {dataset_path.resolve()}"
        )

    df = pd.read_csv(dataset_path, low_memory=False)
    validate_dataset(df)

    X = build_text(df)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    rows = []
    for tree_index, estimator in enumerate(model.estimators_):
        rows.append({
            "arbol": tree_index,
            "profundidad": estimator.tree_.max_depth,
            "nodos": estimator.tree_.node_count,
            "hojas": estimator.tree_.n_leaves,
        })

    summary = pd.DataFrame(rows)
    feature_names = vectorizer.get_feature_names_out()

    selected_index, selected_terms = choose_readable_tree(
        model,
        feature_names,
        summary,
    )

    summary.to_csv(
        output_dir / "resumen_estructura_arboles.csv",
        index=False,
    )

    # FIGURA 1: distribución de profundidades
    median_depth = summary["profundidad"].median()

    plt.figure(figsize=(10, 6))
    plt.hist(summary["profundidad"], bins="auto")
    plt.axvline(
        median_depth,
        linestyle="--",
        linewidth=1.5,
        label=f"Mediana: {median_depth:.1f}",
    )
    plt.xlabel("Profundidad máxima")
    plt.ylabel("Número de árboles")
    plt.title("Distribución de la profundidad de los 200 árboles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "distribucion_profundidad_arboles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # FIGURA 2: árbol ilustrativo con texto más grande
    plt.figure(figsize=(14, 8))
    plot_tree(
        model.estimators_[selected_index],
        max_depth=2,
        feature_names=feature_names,
        class_names=["Normal", "Ataque"],
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=18,
    )
    plt.title(
        "Árbol ilustrativo del modelo Random Forest "
        "(primeros 3 niveles)",
        fontsize=20,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "arbol_random_forest_ilustrativo.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    pd.DataFrame({
        "arbol_seleccionado": [selected_index],
        "terminos_primeros_niveles": [", ".join(selected_terms)],
        "profundidad_completa": [
            model.estimators_[selected_index].tree_.max_depth
        ],
        "numero_nodos": [
            model.estimators_[selected_index].tree_.node_count
        ],
        "numero_hojas": [
            model.estimators_[selected_index].tree_.n_leaves
        ],
    }).to_csv(
        output_dir / "arbol_ilustrativo_seleccionado.csv",
        index=False,
    )

    print("=== RESULTADO ===")
    print("Árbol ilustrativo seleccionado:", selected_index)
    print("Términos de los primeros niveles:", ", ".join(selected_terms))
    print(
        "Profundidad mínima / mediana / máxima:",
        int(summary["profundidad"].min()),
        f"{median_depth:.1f}",
        int(summary["profundidad"].max()),
    )
    print("\nArchivos generados:")
    print("-", output_dir / "distribucion_profundidad_arboles.png")
    print("-", output_dir / "arbol_random_forest_ilustrativo.png")
    print("-", output_dir / "resumen_estructura_arboles.csv")
    print("-", output_dir / "arbol_ilustrativo_seleccionado.csv")


if __name__ == "__main__":
    main()
