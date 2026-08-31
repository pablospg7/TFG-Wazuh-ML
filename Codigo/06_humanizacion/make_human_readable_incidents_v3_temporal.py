
import re
import pandas as pd

INPUT_FILE = "ml_only_attacks_readable.csv"
WINDOW_MINUTES = 5

# =====================================================
# REGEX
# =====================================================
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

APACHE_TS_RE = re.compile(
    r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\]"
)

SYSLOG_TS_RE = re.compile(
    r"\b([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\b"
)

USER_PATTERNS = [
    re.compile(r"\bfor invalid user (\w+)", re.IGNORECASE),
    re.compile(r"\bfor user (\w+)", re.IGNORECASE),
    re.compile(r"\buser[ =:]+(\w+)", re.IGNORECASE),
    re.compile(r"\bdstuser[ =:]+(\w+)", re.IGNORECASE),
    re.compile(r"\bfor (\w+) from\b", re.IGNORECASE),
]

URL_RE = re.compile(
    r"\"(?:GET|POST|HEAD|PUT|DELETE)\s+([^ ]+)\s+HTTP",
    re.IGNORECASE
)

# =====================================================
# CLASIFICACIÓN HUMANA
# =====================================================
def estimate_attack_type(text: str) -> str:

    t = str(text).lower()

    ssh_score = sum(k in t for k in [
        "sshd",
        "failed password",
        "authentication failed",
        "maximum authentication attempts exceeded",
        "preauth",
        "pam",
        "too many authentication failures",
        "retries",
        "ssh",
    ])

    web_score = sum(k in t for k in [
        "gobuster",
        "dirb",
        "get ",
        "http",
        "404",
        "directory",
        "forbidden",
        "suspicious url",
    ])

    successful_ssh = any(k in t for k in [
        "accepted password",
        "session opened",
        "accepted publickey",
    ])

    if successful_ssh:
        return "Posible acceso exitoso tras fuerza bruta SSH"

    if ssh_score > web_score and ssh_score > 0:
        return "Posible fuerza bruta SSH"

    if web_score > ssh_score and web_score > 0:
        return "Posible enumeración web"

    if ssh_score > 0 and web_score > 0:
        return "Actividad mixta SSH/Web"

    return "Actividad sospechosa no clasificada"


# =====================================================
# HERRAMIENTA
# =====================================================
def estimate_tool(text: str, attack_type: str) -> str:

    t = str(text).lower()

    if attack_type == "Posible enumeración web":

        if "gobuster" in t:
            return "Gobuster"

        if "dirb" in t:
            return "Dirb"

        return "Enumeración web no identificada"

    if attack_type == "Posible fuerza bruta SSH":

        if "hydra" in t:
            return "Hydra"

        return "Brute force SSH"

    if attack_type == "Posible acceso exitoso tras fuerza bruta SSH":
        return "Posible acceso SSH exitoso"

    return "No identificado"


# =====================================================
# EXTRACTORES
# =====================================================
def extract_ip(text: str) -> str:

    m = IP_RE.search(str(text))

    return m.group(0) if m else "ip_desconocida"


def extract_user(text: str, attack_type: str) -> str:

    if attack_type not in [
        "Posible fuerza bruta SSH",
        "Posible acceso exitoso tras fuerza bruta SSH"
    ]:
        return ""

    for pattern in USER_PATTERNS:

        m = pattern.search(str(text))

        if m:

            user = m.group(1)

            if user.lower().startswith("http"):
                return ""

            return user

    return ""


def extract_url(text: str) -> str:

    m = URL_RE.search(str(text))

    return m.group(1) if m else ""


# =====================================================
# TIMESTAMPS
# =====================================================
def extract_timestamp(text: str):

    text = str(text)

    # Apache
    m = APACHE_TS_RE.search(text)

    if m:

        ts = pd.to_datetime(
            m.group(1),
            format="%d/%b/%Y:%H:%M:%S %z",
            errors="coerce",
            utc=True
        )

        if pd.notna(ts):
            return ts

    # Syslog
    m = SYSLOG_TS_RE.search(text)

    if m:

        ts = pd.to_datetime(
            "2026 " + m.group(1),
            format="%Y %b %d %H:%M:%S",
            errors="coerce",
            utc=True
        )

        if pd.notna(ts):
            return ts

    # ISO T
    m = re.search(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        text
    )

    if m:

        ts = pd.to_datetime(
            m.group(0),
            errors="coerce",
            utc=True
        )

        if pd.notna(ts):
            return ts

    # ISO SPACE
    m = re.search(
        r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
        text
    )

    if m:

        ts = pd.to_datetime(
            m.group(0),
            errors="coerce",
            utc=True
        )

        if pd.notna(ts):
            return ts

    return pd.NaT


# =====================================================
# RESUMEN HUMANO
# =====================================================
def make_summary(row: pd.Series) -> str:

    attack_type = row["tipo_ataque_humano"]
    ip = row["ip_origen"]
    user = row["usuario_objetivo"]
    tool = row["herramienta_estimada"]
    num = row["num_eventos"]
    duration = row["duracion_minutos"]
    urls = row["urls_ejemplo"]

    if pd.notna(duration):
        duration_txt = (
            f" durante aproximadamente {duration:.1f} minutos"
        )
    else:
        duration_txt = ""

    # =================================================
    # ACCESO EXITOSO SSH
    # =================================================
    if attack_type == "Posible acceso exitoso tras fuerza bruta SSH":

        if user:
            return (
                f"{attack_type} desde {ip} "
                f"sobre el usuario {user}, "
                f"con {num} eventos correlacionados"
                f"{duration_txt}. "
                f"Esto podría indicar un compromiso "
                f"exitoso del sistema."
            )

        return (
            f"{attack_type} desde {ip}, "
            f"con {num} eventos correlacionados"
            f"{duration_txt}. "
            f"Esto podría indicar un compromiso "
            f"exitoso del sistema."
        )

    # =================================================
    # FUERZA BRUTA SSH
    # =================================================
    if attack_type == "Posible fuerza bruta SSH":

        if user:
            return (
                f"{attack_type} desde {ip} "
                f"contra el usuario {user}, "
                f"con {num} eventos correlacionados"
                f"{duration_txt}. "
                f"Herramienta estimada: {tool}."
            )

        return (
            f"{attack_type} desde {ip}, "
            f"con {num} eventos correlacionados"
            f"{duration_txt}. "
            f"Herramienta estimada: {tool}."
        )

    # =================================================
    # WEB
    # =================================================
    if attack_type == "Posible enumeración web":

        if urls:
            return (
                f"{attack_type} desde {ip}, "
                f"con {num} peticiones sospechosas "
                f"correlacionadas{duration_txt}. "
                f"Herramienta estimada: {tool}. "
                f"Recursos de ejemplo: {urls}."
            )

        return (
            f"{attack_type} desde {ip}, "
            f"con {num} peticiones sospechosas "
            f"correlacionadas{duration_txt}. "
            f"Herramienta estimada: {tool}."
        )

    # =================================================
    # MIXTO
    # =================================================
    if attack_type == "Actividad mixta SSH/Web":

        return (
            f"{attack_type} desde {ip}, "
            f"con {num} eventos correlacionados"
            f"{duration_txt}. "
            f"Herramienta estimada: {tool}."
        )

    return (
        f"{attack_type} desde {ip}, "
        f"con {num} eventos sospechosos "
        f"correlacionados{duration_txt}."
    )


# =====================================================
# MAIN
# =====================================================
df = pd.read_csv(INPUT_FILE, low_memory=False)

if "log" not in df.columns:
    raise ValueError("No existe la columna 'log'.")

df["log_text"] = df["log"].fillna("").astype(str)

# =====================================================
# EXTRAER CAMPOS
# =====================================================
df["tipo_ataque_humano"] = df["log_text"].apply(
    estimate_attack_type
)

df["ip_origen"] = df["log_text"].apply(
    extract_ip
)

df["herramienta_estimada"] = df.apply(
    lambda row: estimate_tool(
        row["log_text"],
        row["tipo_ataque_humano"]
    ),
    axis=1
)

df["usuario_objetivo"] = df.apply(
    lambda row: extract_user(
        row["log_text"],
        row["tipo_ataque_humano"]
    ),
    axis=1
)

df["url"] = df["log_text"].apply(
    extract_url
)

df["timestamp"] = df["log_text"].apply(
    extract_timestamp
)

# =====================================================
# TIME BUCKET
# =====================================================
if df["timestamp"].notna().any():

    df["time_bucket"] = df["timestamp"].dt.floor(
        f"{WINDOW_MINUTES}min"
    )

else:
    df["time_bucket"] = "sin_timestamp"

# =====================================================
# AGRUPACIÓN
# =====================================================
group_cols = [
    "tipo_ataque_humano",
    "ip_origen",
    "usuario_objetivo",
    "herramienta_estimada",
    "time_bucket",
]


def top_urls(series):

    urls = []

    for x in series:

        if (
            isinstance(x, str)
            and x.strip()
            and x not in urls
        ):
            urls.append(x)

    return ", ".join(urls[:5])


grouped = (
    df.groupby(group_cols, dropna=False)
    .agg(
        num_eventos=("log_text", "size"),
        primer_evento=("timestamp", "min"),
        ultimo_evento=("timestamp", "max"),
        ejemplo_log=("log_text", "first"),
        urls_ejemplo=("url", top_urls),
    )
    .reset_index()
)

# =====================================================
# DURACIÓN
# =====================================================
grouped["duracion_minutos"] = (
    (
        grouped["ultimo_evento"]
        - grouped["primer_evento"]
    ).dt.total_seconds()
    / 60
)

# =====================================================
# RESUMEN HUMANO
# =====================================================
grouped["resumen_humano"] = grouped.apply(
    make_summary,
    axis=1
)

# Ordenar
grouped = grouped.sort_values(
    "num_eventos",
    ascending=False
)

# =====================================================
# GUARDAR CSV
# =====================================================
grouped.to_csv(
    "ml_incidents_human_readable_v3_temporal.csv",
    index=False
)

# =====================================================
# GUARDAR TXT
# =====================================================
with open(
    "ml_incidents_human_readable_v3_temporal.txt",
    "w",
    encoding="utf-8"
) as f:

    for _, row in grouped.iterrows():

        f.write(row["resumen_humano"] + "\n")

        f.write(
            f"  - Tipo: "
            f"{row['tipo_ataque_humano']}\n"
        )

        f.write(
            f"  - IP origen: "
            f"{row['ip_origen']}\n"
        )

        if row["usuario_objetivo"]:

            f.write(
                f"  - Usuario objetivo: "
                f"{row['usuario_objetivo']}\n"
            )

        f.write(
            f"  - Herramienta estimada: "
            f"{row['herramienta_estimada']}\n"
        )

        f.write(
            f"  - Eventos correlacionados: "
            f"{row['num_eventos']}\n"
        )

        f.write(
            f"  - Inicio: "
            f"{row['primer_evento']}\n"
        )

        f.write(
            f"  - Fin: "
            f"{row['ultimo_evento']}\n"
        )

        if row["urls_ejemplo"]:

            f.write(
                f"  - URLs ejemplo: "
                f"{row['urls_ejemplo']}\n"
            )

        f.write(
            f"  - Ejemplo log: "
            f"{row['ejemplo_log'][:300]}\n"
        )

        f.write("\n")

# =====================================================
# RESUMEN FINAL
# =====================================================
print("Archivos generados:")
print("- ml_incidents_human_readable_v3_temporal.csv")
print("- ml_incidents_human_readable_v3_temporal.txt")

print("\nResumen:")
print("Eventos analizados:", len(df))
print("Incidentes agrupados:", len(grouped))

print("\nTipos de incidentes:")
print(
    grouped["tipo_ataque_humano"].value_counts()
)

print("\nPrimeros incidentes:")

for s in grouped["resumen_humano"].head(10):
    print("-", s)
