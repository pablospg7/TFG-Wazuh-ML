import pandas as pd

files = [
    "wazuh_alerts_ssh_raw_labeled.csv",
    "web_attack_dirb_gobuster_alerts_labeled.csv",
    "web_normal_alertas_labeled.csv",
    "AlertasNormal_labeled.csv"
    
]

dfs = []

for f in files:
    print("Loading:", f)
    df = pd.read_csv(f)
    dfs.append(df)

alerts = pd.concat(dfs, ignore_index=True)

print("\nTotal alerts:", len(alerts))
print(alerts["label"].value_counts())

alerts.to_csv("wazuh_alerts_all.csv", index=False)

print("\nSaved wazuh_alerts_all.csv")