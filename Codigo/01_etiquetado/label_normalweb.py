import pandas as pd

# ARCHIVES
df = pd.read_csv("web_normal_archives.csv")
df["label"] = 0
df["label_reason"] = "normal_web"
df.to_csv("web_normal_archives_labeled.csv", index=False)

# ALERTS
df = pd.read_csv("web_normal_alertas.csv")
df["label"] = 0
df["label_reason"] = "normal_web"
df.to_csv("web_normal_alertas_labeled.csv", index=False)

print("Web normal etiquetado correctamente")