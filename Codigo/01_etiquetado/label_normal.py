import pandas as pd
from pathlib import Path

ARCHIVE_PATH = Path("ArchiveNormal.csv")
ALERTS_PATH  = Path("AlertasNormal.csv")

def load_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {path.resolve()}")

    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)


def add_labels_and_save(df: pd.DataFrame, label: int, dataset_name: str, out_path: Path):
    df = df.copy()
    df["label"] = label
    df["dataset"] = dataset_name
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] {out_path.name} -> filas={len(df)} columnas={df.shape[1]}")


def main():
    # NORMAL - ARCHIVES
    df_arch = load_csv_safely(ARCHIVE_PATH)
    add_labels_and_save(
        df_arch,
        label=0,
        dataset_name="normal_archive",
        out_path=Path("ArchiveNormal_labeled.csv")
    )

    # NORMAL - ALERTS
    df_alert = load_csv_safely(ALERTS_PATH)
    add_labels_and_save(
        df_alert,
        label=0,
        dataset_name="normal_alerts",
        out_path=Path("AlertasNormal_labeled.csv")
    )

    print("\n✔ Dataset NORMAL preparado sin eliminar duplicados.")

if __name__ == "__main__":
    main()