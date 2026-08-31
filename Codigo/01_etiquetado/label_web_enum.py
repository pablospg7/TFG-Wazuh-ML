import argparse
import re
import pandas as pd

# -------------------------
# Helpers
# -------------------------
STATUS_RE = re.compile(r'"\s*(\d{3})\s')  # captura status code típico en access logs
UA_KEYWORDS = ["gobuster", "dirb", "dirbuster", "ffuf", "nikto", "wpscan"]
WEB_ENUM_RULE_IDS = {31101, 31102, 31103, 31104, 31105}  # típicos web error codes en Wazuh (puedes ampliar)

def find_timestamp_col(df: pd.DataFrame):
    # En exports de Wazuh suele venir alguno de estos
    for c in ["_source.@timestamp", "@timestamp", "timestamp", "_source.timestamp"]:
        if c in df.columns:
            return c
    return None

def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def extract_status_from_full_log(series: pd.Series) -> pd.Series:
    # Devuelve status code (int) o NaN
    def _one(s):
        if not isinstance(s, str):
            return None
        m = STATUS_RE.search(s)
        return int(m.group(1)) if m else None
    return series.apply(_one)

def to_bool_contains(series: pd.Series, keywords):
    s = series.astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | s.str.contains(re.escape(kw), na=False)
    return mask

# -------------------------
# Core labeling
# -------------------------
def label_web_csv(
    input_csv: str,
    output_csv: str,
    attacker_ips: list[str],
    window_seconds: int,
    min_requests_in_window: int,
    min_unique_urls_in_window: int,
):
    df = pd.read_csv(input_csv)

    # columnas típicas
    rule_id_col = col(df, "_source.rule.id")
    rule_desc_col = col(df, "_source.rule.description")
    rule_groups_col = col(df, "_source.rule.groups")
    srcip_col = col(df, "_source.data.srcip", "_source.srcip", "srcip")
    url_col = col(df, "_source.data.url", "_source.url", "url")
    full_log_col = col(df, "_source.full_log", "full_log", "message")
    ts_col = find_timestamp_col(df)

    # Normalizaciones seguras
    if rule_id_col:
        df["rule_id"] = pd.to_numeric(df[rule_id_col], errors="coerce")
    else:
        df["rule_id"] = pd.NA

    df["rule_desc_lc"] = df[rule_desc_col].astype(str).str.lower() if rule_desc_col else ""
    df["groups_lc"] = df[rule_groups_col].astype(str).str.lower() if rule_groups_col else ""
    df["srcip"] = df[srcip_col].astype(str) if srcip_col else ""
    df["url"] = df[url_col].astype(str) if url_col else ""

    if full_log_col:
        df["full_log_lc"] = df[full_log_col].astype(str).str.lower()
        df["status_code"] = extract_status_from_full_log(df[full_log_col])
    else:
        df["full_log_lc"] = ""
        df["status_code"] = pd.NA

    # -------------------------
    # 1) Alta confianza (label_hc)
    # -------------------------
    is_attacker_ip = df["srcip"].isin(attacker_ips) if attacker_ips else False

    # Si Wazuh ya lo mete en grupo "attack", lo aprovechamos
    group_says_attack = df["groups_lc"].str.contains("attack", na=False)

    # Reglas típicas web (errores 4xx) + atacante IP
    rule_is_web_enum = df["rule_id"].isin(WEB_ENUM_RULE_IDS)

    # status 4xx desde full_log (por si el CSV no trae rule.id consistente)
    status_4xx = pd.to_numeric(df["status_code"], errors="coerce").between(400, 499, inclusive="both")

    # keywords de herramientas en user-agent / full_log
    tool_kw = to_bool_contains(df["full_log_lc"], UA_KEYWORDS)

    # Alta confianza: (grupo attack) OR (attacker_ip AND (rule_web_enum OR status_4xx OR tool_kw))
    df["label_hc"] = (group_says_attack | (is_attacker_ip & (rule_is_web_enum | status_4xx | tool_kw))).astype(int)

    # -------------------------
    # 2) Correlación temporal (label_corr)
    #    Idea: ráfaga de 4xx desde misma IP + muchos paths distintos en una ventana corta
    # -------------------------
    df["label_corr"] = 0

    if ts_col:
        # parse timestamp
        df["ts"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        # filtramos filas con ts válida y srcip
        work = df.dropna(subset=["ts"]).copy()

        # solo consideramos 4xx para correlación (más típico de enumeración)
        work["is_4xx"] = pd.to_numeric(work["status_code"], errors="coerce").between(400, 499, inclusive="both")
        work = work[work["is_4xx"]].copy()

        if len(work) > 0:
            # Ventanizamos por "bucket" de window_seconds
            # (redondeo hacia abajo)
            bucket = (work["ts"].astype("int64") // (window_seconds * 1_000_000_000))  # ns -> bucket
            work["bucket"] = bucket

            # agregados por srcip+bucket
            agg = (
                work.groupby(["srcip", "bucket"])
                .agg(reqs=("url", "size"), unique_urls=("url", pd.Series.nunique))
                .reset_index()
            )

            suspicious_bins = agg[
                (agg["reqs"] >= min_requests_in_window) &
                (agg["unique_urls"] >= min_unique_urls_in_window)
            ][["srcip", "bucket"]]

            if len(suspicious_bins) > 0:
                work = work.merge(suspicious_bins.assign(_corr=1), on=["srcip", "bucket"], how="left")
                corr_idx = work.index[work["_corr"].fillna(0).astype(int) == 1]
                df.loc[corr_idx, "label_corr"] = 1

    # -------------------------
    # 3) Label final + razón
    # -------------------------
    df["label"] = ((df["label_hc"] == 1) | (df["label_corr"] == 1)).astype(int)

    def reason_row(r):
        if r["label_hc"] == 1 and r["label_corr"] == 1:
            return "hc+correlation"
        if r["label_hc"] == 1:
            if "attack" in str(r.get("groups_lc", "")):
                return "group_attack"
            if pd.notna(r.get("rule_id")) and float(r["rule_id"]) in WEB_ENUM_RULE_IDS:
                return f"rule_id_web({int(r['rule_id'])})"
            if str(r.get("status_code", "")).isdigit() and 400 <= int(r["status_code"]) <= 499:
                return "status_4xx"
            if any(k in str(r.get("full_log_lc", "")) for k in UA_KEYWORDS):
                return "tool_keyword"
            return "hc_other"
        if r["label_corr"] == 1:
            return "burst_correlation"
        return "normal_or_not_confirmed"

    df["label_reason"] = df.apply(reason_row, axis=1)

    # Resumen
    print("\n=== RESUMEN:", input_csv, "===")
    print("Filas:", len(df))
    print("label_hc=1:", int(df["label_hc"].sum()))
    print("label_corr=1:", int(df["label_corr"].sum()))
    print("label final=1:", int(df["label"].sum()))
    print("Top rule.description (10):")
    if rule_desc_col:
        print(df[rule_desc_col].value_counts().head(10))
    else:
        print("(no hay columna de descripción)")

    df.to_csv(output_csv, index=False)
    print("✅ Guardado:", output_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", help="CSV de alerts (opcional)")
    ap.add_argument("--archives", help="CSV de archives (opcional)")
    ap.add_argument("--attacker-ip", action="append", default=[], help="IP atacante (repetible). Ej: --attacker-ip 192.168.1.50")
    ap.add_argument("--window", type=int, default=60, help="Ventana de correlación (segundos). Default 60")
    ap.add_argument("--min-req", type=int, default=40, help="Mín. peticiones 4xx en la ventana para marcar correlación. Default 40")
    ap.add_argument("--min-urls", type=int, default=20, help="Mín. URLs distintas en la ventana para marcar correlación. Default 20")
    args = ap.parse_args()

    if not args.alerts and not args.archives:
        raise SystemExit("Debes pasar --alerts y/o --archives")

    if args.alerts:
        label_web_csv(
            input_csv=args.alerts,
            output_csv=args.alerts.replace(".csv", "_labeled.csv"),
            attacker_ips=args.attacker_ip,
            window_seconds=args.window,
            min_requests_in_window=args.min_req,
            min_unique_urls_in_window=args.min_urls,
        )

    if args.archives:
        label_web_csv(
            input_csv=args.archives,
            output_csv=args.archives.replace(".csv", "_labeled.csv"),
            attacker_ips=args.attacker_ip,
            window_seconds=args.window,
            min_requests_in_window=args.min_req,
            min_unique_urls_in_window=args.min_urls,
        )

if __name__ == "__main__":
    main()