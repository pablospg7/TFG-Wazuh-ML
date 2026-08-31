#!/usr/bin/env python3
"""
Prototipo TFG:
SHAP -> dataset -> ruleset Wazuh -> reglas candidatas.

No entrena modelos ni recalcula SHAP.
La comparación con las reglas de Wazuh es una ayuda para detectar
posibles redundancias; la validación final debe ser manual.
"""

import html
import re
import tarfile
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# ARCHIVOS
# ------------------------------------------------------------

DATASET_FILE = "dataset_final.csv"
SHAP_FILE = "top_shap_features_rf_clean.csv"
RULESET_FILE = "wazuh_rules_export_20260810_155455.tar.gz"

REPORT_FILE = "shap_wazuh_rule_report.csv"
OUTPUT_XML = "candidate_rules_shap.xml"

TEXT_COLS = [
    "_source.full_log",
    "_source.data.url",
    "_source.data.dstuser",
]

TERM_COL = "feature"
SHAP_COL = "mean_abs_shap"

MIN_APPEARANCES = 10
MIN_ATTACK_RATIO = 0.90
MAX_NORMAL = 5
START_RULE_ID = 110200

# Palabras demasiado generales o propias del laboratorio.
EXCLUDED_TERMS = {
    "get", "post", "http", "https", "port",
    "for", "from", "failed", "password",
    "user", "root", "ssh", "sshd", "pam",
    "service", "session", "error",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}

# Campos de una regla Wazuh que pueden contener condiciones.
RULE_TAGS = [
    "match", "regex", "field",
    "srcip", "dstip", "srcport", "dstport",
    "srcuser", "dstuser", "user",
    "url", "id", "status",
    "program_name", "hostname", "location",
    "decoded_as", "category",
]


def build_text(df):
    """Une los mismos campos textuales usados por el modelo."""
    usable = [c for c in TEXT_COLS if c in df.columns]

    if not usable:
        raise ValueError("No se encontraron campos textuales.")

    text = pd.Series("", index=df.index, dtype="object")

    for col in usable:
        text += " " + df[col].fillna("").astype(str)

    return text.str.lower()


def make_regex(term):
    return rf"(?i)\b{re.escape(term)}\b"


def clean_xml_text(value):
    """Quita etiquetas XML y normaliza espacios."""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def get_tag_values(rule_xml, tag):
    pattern = rf"<{tag}\b[^>]*>(.*?)</{tag}>"

    return [
        clean_xml_text(value)
        for value in re.findall(
            pattern,
            rule_xml,
            flags=re.I | re.S,
        )
    ]


def load_wazuh_rules():
    """Lee las reglas XML directamente desde el .tar.gz."""
    rules = []

    with tarfile.open(RULESET_FILE, "r:gz") as tar:
        xml_files = [
            member
            for member in tar.getmembers()
            if member.isfile()
            and member.name.lower().endswith(".xml")
        ]

        for member in xml_files:
            file_obj = tar.extractfile(member)

            if file_obj is None:
                continue

            content = file_obj.read().decode(
                "utf-8",
                errors="ignore",
            )

            blocks = re.findall(
                r"<rule\b[^>]*>.*?</rule>",
                content,
                flags=re.I | re.S,
            )

            for block in blocks:
                opening = re.search(
                    r"<rule\b([^>]*)>",
                    block,
                    flags=re.I,
                )

                if not opening:
                    continue

                attrs = opening.group(1)

                rule_id = re.search(
                    r'\bid\s*=\s*["\'](\d+)["\']',
                    attrs,
                    flags=re.I,
                )

                level = re.search(
                    r'\blevel\s*=\s*["\'](\d+)["\']',
                    attrs,
                    flags=re.I,
                )

                if not rule_id:
                    continue

                descriptions = get_tag_values(
                    block,
                    "description",
                )

                conditions = []

                for tag in RULE_TAGS:
                    conditions.extend(
                        get_tag_values(block, tag)
                    )

                rules.append({
                    "rule_id": int(rule_id.group(1)),
                    "level": (
                        int(level.group(1))
                        if level
                        else 0
                    ),
                    "file": member.name,
                    "description": " ".join(descriptions),
                    "conditions": " ".join(conditions).lower(),
                })

    return rules


def contains_term(text, term):
    pattern = (
        rf"(?<![a-z0-9_])"
        rf"{re.escape(term.lower())}"
        rf"(?![a-z0-9_])"
    )

    return bool(re.search(pattern, text.lower()))


def find_rules(term, rules):
    """Devuelve las reglas que contienen el término."""
    return [
        rule
        for rule in rules
        if contains_term(rule["conditions"], term)
    ]


def best_existing_rule(hits, useful_terms):
    """
    Prioriza la regla que contiene más términos SHAP
    considerados específicos de ataque.
    """
    if not hits:
        return None, []

    best_rule = None
    best_terms = []

    for rule in hits:
        matched = [
            term
            for term in useful_terms
            if contains_term(rule["conditions"], term)
        ]

        if len(matched) > len(best_terms):
            best_rule = rule
            best_terms = matched

    return best_rule, best_terms


def free_rule_ids(rules, amount=2):
    """Busca IDs libres para las reglas locales candidatas."""
    used = {rule["rule_id"] for rule in rules}

    result = []
    candidate = START_RULE_ID

    while len(result) < amount:
        if candidate not in used:
            result.append(candidate)

        candidate += 1

    return result


def write_candidate_xml(candidates, rules):
    """Genera XML únicamente para candidatos no cubiertos."""
    lines = [
        '<group name="local,ml_shap_candidates,">',
        "",
        "  <!-- Prototipo TFG: revisar antes de utilizar. -->",
        "",
    ]

    next_ids = free_rule_ids(
        rules,
        amount=max(2, len(candidates) * 2),
    )
    pos = 0

    for term in candidates["term"]:
        # Para Gobuster se usa correlación para no generar
        # una alerta por cada petición individual.
        if term == "gobuster":
            base_id = next_ids[pos]
            corr_id = next_ids[pos + 1]
            pos += 2

            lines += [
                f'  <rule id="{base_id}" level="1">',
                "    <if_sid>31100</if_sid>",
                '    <match type="pcre2">'
                '(?i)\\bgobuster(?:/[0-9.]+)?\\b'
                "</match>",
                "    <description>"
                "TFG: Gobuster request observed."
                "</description>",
                "    <options>no_log</options>",
                "    <group>web,reconnaissance,gobuster,</group>",
                "  </rule>",
                "",
                (
                    f'  <rule id="{corr_id}" level="10" '
                    'frequency="10" timeframe="60">'
                ),
                f"    <if_matched_sid>{base_id}</if_matched_sid>",
                "    <same_srcip />",
                "    <description>"
                "TFG: Possible web enumeration using Gobuster."
                "</description>",
                "    <group>"
                "attack,web_scan,reconnaissance,gobuster,"
                "</group>",
                "  </rule>",
                "",
            ]

        else:
            rule_id = next_ids[pos]
            pos += 1

            lines += [
                f'  <rule id="{rule_id}" level="8">',
                (
                    '    <match type="pcre2">'
                    f'{html.escape(make_regex(term))}'
                    "</match>"
                ),
                (
                    "    <description>"
                    f"TFG: SHAP candidate - {html.escape(term)}"
                    "</description>"
                ),
                "    <group>attack,ml_shap_candidate,</group>",
                "  </rule>",
                "",
            ]

    lines.append("</group>")

    Path(OUTPUT_XML).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    # 1. Cargar dataset y SHAP.
    dataset = pd.read_csv(
        DATASET_FILE,
        low_memory=False,
    )

    shap_df = pd.read_csv(SHAP_FILE)

    if "label" not in dataset.columns:
        raise ValueError("Falta la columna label.")

    if not {TERM_COL, SHAP_COL}.issubset(shap_df.columns):
        raise ValueError(
            "El CSV SHAP necesita feature y mean_abs_shap."
        )

    text = build_text(dataset)
    labels = dataset["label"].astype(int)

    # 2. Cargar las reglas reales de Wazuh.
    rules = load_wazuh_rules()

    print(f"Reglas Wazuh leídas: {len(rules)}")

    # 3. Contrastar cada término SHAP con el dataset.
    results = []

    for _, row in shap_df.iterrows():
        term = str(row[TERM_COL]).lower().strip()

        mask = text.str.contains(
            make_regex(term),
            regex=True,
            na=False,
        )

        attacks = int(((labels == 1) & mask).sum())
        normals = int(((labels == 0) & mask).sum())
        total = attacks + normals

        ratio = attacks / total if total else 0

        passes_dataset = (
            total >= MIN_APPEARANCES
            and ratio >= MIN_ATTACK_RATIO
            and normals <= MAX_NORMAL
        )

        results.append({
            "term": term,
            "mean_abs_shap": float(row[SHAP_COL]),
            "attacks": attacks,
            "normals": normals,
            "total": total,
            "attack_ratio": round(ratio, 4),
            "passes_dataset": passes_dataset,
        })

    # Términos suficientemente específicos según el dataset.
    useful_terms = [
        row["term"]
        for row in results
        if row["passes_dataset"]
        and row["term"] not in EXCLUDED_TERMS
    ]

    # 4. Comparar con el ruleset.
    final = []

    for row in results:
        term = row["term"]

        hits = find_rules(term, rules)

        best_rule, matched_terms = best_existing_rule(
            hits,
            useful_terms,
        )

        if term in EXCLUDED_TERMS:
            status = "DESCARTADO_GENERICO"

        elif not row["passes_dataset"]:
            status = "DESCARTADO_DATASET"

        elif best_rule and len(matched_terms) >= 2:
            status = "REDUNDANCIA_PROBABLE"

        elif hits:
            status = "REVISAR_COBERTURA"

        else:
            status = "CANDIDATO"

        final.append({
            **row,
            "ruleset_hits": len(hits),
            "best_rule_id": (
                best_rule["rule_id"]
                if best_rule
                else ""
            ),
            "best_rule_description": (
                best_rule["description"]
                if best_rule
                else ""
            ),
            "shap_terms_in_best_rule": (
                ", ".join(matched_terms)
                if matched_terms
                else ""
            ),
            "status": status,
        })

    report = pd.DataFrame(final)

    report.to_csv(
        REPORT_FILE,
        index=False,
    )

    # 5. Generar XML solo para candidatos nuevos.
    candidates = report[
        report["status"] == "CANDIDATO"
    ]

    write_candidate_xml(
        candidates,
        rules,
    )

    # 6. Mostrar resultado.
    print("\n=== Resultado ===")
    print(report["status"].value_counts())
    print()

    print("Candidatos nuevos:")

    if candidates.empty:
        print("Ninguno.")
    else:
        print(
            candidates[
                [
                    "term",
                    "attacks",
                    "normals",
                    "attack_ratio",
                ]
            ].to_string(index=False)
        )

    print()
    print(f"Informe: {REPORT_FILE}")
    print(f"XML: {OUTPUT_XML}")
    print()
    print(
        "Las coincidencias con el ruleset son orientativas. "
        "Las reglas candidatas deben revisarse y probarse "
        "con wazuh-logtest."
    )


if __name__ == "__main__":
    main()
