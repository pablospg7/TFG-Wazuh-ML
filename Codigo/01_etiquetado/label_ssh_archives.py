import argparse
import re
import pandas as pd

# -------------------------
# Column helpers
# -------------------------
def pick_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_timestamp_col(df):
    # frecuentes en exports Wazuh/Kibana
    return pick_existing(df, ["_source.@timestamp", "@timestamp", "timestamp", "_source.timestamp"])

def to_lc(series):
    return series.astype(str).str.lower()

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

# -------------------------
# Labeling core
# -------------------------
def label_ssh(
    input_csv: str,
    output_csv: str,
    window_seconds: int,
    threshold: int,
    require_same_user: bool = False,
):
    df = pd.read_csv(input_csv)

    # columnas típicas (archives/alerts)
    rule_id_col = pick_existing(df, ["_source.rule.id", "rule.id", "rule_id"])
    rule_desc_col = pick_existing(df, ["_source.rule.description", "rule.description", "rule_description"])
    full_log_col = pick_existing(df, ["_source.full_log", "full_log", "message"])
    srcip_col = pick_existing(df, ["_source.data.srcip", "_source.srcip", "srcip", "data.srcip"])
    dstuser_col = pick_existing(df, ["_source.data.dstuser", "dstuser", "data.dstuser"])
    ts_col = find_timestamp_col(df)

    missing = []
    if ts_col is None: missing.append("timestamp")
    if srcip_col is None: missing.append("srcip")
    if rule_desc_col is None and full_log_col is None: missing.append("rule.description OR full_log")

    if missing:
        raise SystemExit(
            f"Faltan columnas necesarias: {missing}\n"
            f"Columnas disponibles (primeras 80): {list(df.columns)[:80]}"
        )

    # Normalizar textos
    df["srcip"] = df[srcip_col].astype(str)
    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    # Texto principal para keywords (mejor usar rule.description si existe; si no full_log)
    if rule_desc_col is not None:
        df["text_lc"] = to_lc(df[rule_desc_col])
    else:
        df["text_lc"] = to_lc(df[full_log_col])

    # rule_id num
    if rule_id_col is not None:
        df["rule_id_num"] = pd.to_numeric(df[rule_id_col], errors="coerce")
    else:
        df["rule_id_num"] = pd.NA

    # user (opcional)
    if dstuser_col is not None:
        df["dstuser"] = df[dstuser_col].astype(str)
    else:
        df["dstuser"] = ""

    # -------------------------
    # 1) Alta confianza
    # -------------------------
    # Ajusta aquí si quieres ser más estricto
    ATTACK_RULE_IDS = {
        110100,  # tu regla custom “Confirmed SSH brute force”
        5763,    # sshd brute force trying...
        5758,    # Maximum authentication attempts exceeded
        40111,   # Multiple authentication failures
    }

    STRONG_KEYWORDS = [
        "brute force",
        "maximum authentication attempts exceeded",
        "multiple authentication failures",
        "confirmed ssh brute force",
    ]

    df["label_hc"] = 0
    cond_id = df["rule_id_num"].isin(ATTACK_RULE_IDS)
    cond_kw = False
    for kw in STRONG_KEYWORDS:
        cond_kw = cond_kw | df["text_lc"].str.contains(re.escape(kw), na=False)
    df["label_hc"] = (cond_id | cond_kw).astype(int)

    # -------------------------
    # 2) Correlación temporal sobre eventos débiles
    # -------------------------
    WEAK_KEYWORDS = [
        "authentication failed",
        "failed password",
        "invalid user",
        "preauth",
        "pam: user login failed",
        "user authentication failure",
        "missed the password",
    ]

    df["is_weak"] = False
    for kw in WEAK_KEYWORDS:
        df["is_weak"] = df["is_weak"] | df["text_lc"].str.contains(re.escape(kw), na=False)

    df["label_corr"] = 0
    df["corr_count_window"] = pd.NA

    # trabajar solo con filas con ts válida
    work = df[df["ts"].notna()].copy()
    # solo weak
    weak = work[work["is_weak"]].copy()
    if len(weak) > 0:
        weak = weak.sort_values(["srcip", "ts"])
        weak = weak.set_index("ts")

        group_cols = ["srcip"]
        if require_same_user and "dstuser" in weak.columns:
            group_cols = ["srcip", "dstuser"]

        # rolling por ventana temporal
        rolling_counts = (
            weak.groupby(group_cols)
                .rolling(f"{window_seconds}s")["is_weak"]
                .count()
                .reset_index(name="weak_count_window")
        )

        weak = weak.reset_index()
        weak = weak.merge(rolling_counts, on=(group_cols + ["ts"]), how="left")

        weak["label_corr"] = (weak["weak_count_window"] >= threshold).astype(int)

        # volcamos al df original por índice (usamos el index original antes de reset)
        # Truco: guardamos idx original
        weak["_orig_idx"] = weak.index
        df.loc[weak["_orig_idx"], "label_corr"] = weak["label_corr"].values
        df.loc[weak["_orig_idx"], "corr_count_window"] = weak["weak_count_window"].values

    # -------------------------
    # 3) Label final + razones
    # -------------------------
    df["label"] = ((df["label_hc"] == 1) | (df["label_corr"] == 1)).astype(int)

    def reason_row(r):
        if r["label_hc"] == 1 and r["label_corr"] == 1:
            return "hc+correlation"
        if r["label_hc"] == 1:
            rid = safe_int(r.get(rule_id_col)) if rule_id_col else None
            if rid in ATTACK_RULE_IDS:
                return f"hc_rule_id({rid})"
            for kw in STRONG_KEYWORDS:
                if kw in str(r.get("text_lc", "")):
                    return f"hc_keyword({kw})"
            return "hc_other"
        if r["label_corr"] == 1:
            return f"burst_corr({r.get('corr_count_window')} in {window_seconds}s >= {threshold})"
        return "benign_or_unconfirmed"

    df["label_reason"] = df.apply(reason_row, axis=1)

    # -------------------------
    # Summary + Save
    # -------------------------
    print("\n=== RESUMEN SSH ARCHIVES ===")
    print("Archivo:", input_csv)
    print("Filas:", len(df))
    print("label_hc=1:", int(df["label_hc"].sum()))
    print("label_corr=1:", int(df["label_corr"].sum()))
    print("label final=1:", int(df["label"].sum()))
    print("label final=0:", int((df["label"] == 0).sum()))

    if rule_id_col:
        print("\nTop rule.id (10):")
        print(df[rule_id_col].value_counts().head(10))

    if rule_desc_col:
        print("\nTop rule.description (10):")
        print(df[rule_desc_col].value_counts().head(10))

    df.to_csv(output_csv, index=False)
    print("\n✅ Guardado:", output_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV exportado (ARCHIVES o ALERTS)")
    ap.add_argument("--output", default=None, help="CSV de salida (por defecto añade _labeled)")
    ap.add_argument("--window", type=int, default=60, help="Ventana correlación (segundos). Default 60")
    ap.add_argument("--threshold", type=int, default=6, help="Umbral eventos weak en ventana. Default 6")
    ap.add_argument("--require-same-user", action="store_true", help="Si quieres correlación por IP+usuario (más estricto)")
    args = ap.parse_args()

    output = args.output
    if output is None:
        output = args.input.replace(".csv", "_labeled.csv")

    label_ssh(
        input_csv=args.input,
        output_csv=output,
        window_seconds=args.window,
        threshold=args.threshold,
        require_same_user=args.require_same_user,
    )

if __name__ == "__main__":
    main()