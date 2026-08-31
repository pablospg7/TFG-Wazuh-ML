import pandas as pd

files = [
    "ArchiveNormal_labeled.csv",
    "web_normal_archives_labeled.csv",
    "archivessshbueno_labeled.csv",
    "web_attack_dirb_gobuster_archives_labeled.csv"
]

dfs = []

for file in files:
    print("Leyendo:", file)
    df = pd.read_csv(file)

    df["source_dataset"] = file
    dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)

print("\nFilas totales:", len(dataset))
print("\nDistribución labels:")
print(dataset["label"].value_counts())

dataset.to_csv("dataset_final.csv", index=False)

print("\nDataset final guardado como dataset_final.csv")