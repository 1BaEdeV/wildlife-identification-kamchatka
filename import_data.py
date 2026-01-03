"""
pip install pandas requests tqdm
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = "observations-663836.csv"    # ваш файл
OUT_DIR = Path("inat_from_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# какие лицензии допускаем (по колонке license)
ALLOWED_LICENSES = {"CC-BY", "CC0", "CC-BY-SA"}  # при желании расширить

def safe_name(s: str, maxlen: int = 160) -> str:
    import re
    s = re.sub(r"[^\w\-.]+", "_", s).strip("_")
    return s[:maxlen]

def download(url: str, dst: Path, timeout: int = 60):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)

def main():
    df = pd.read_csv(CSV_PATH)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row.get("image_url")
        if not isinstance(img_url, str) or not img_url:
            continue

        license_code = str(row.get("license", "")).strip()
        if ALLOWED_LICENSES and license_code not in ALLOWED_LICENSES:
            continue

        taxon_id = row.get("taxon_id")
        sci_name = row.get("scientific_name", "unknown")
        obs_id = row.get("id")

        # папка по виду
        species_dir = OUT_DIR / f"{taxon_id}__{safe_name(str(sci_name))}"

        ext = os.path.splitext(urlparse(img_url).path)[1].lower() or ".jpg"
        fname = f"obs{obs_id}{ext}"
        dst = species_dir / safe_name(fname)

        status, err = "ok", ""
        if not dst.exists():
            try:
                download(img_url, dst)
            except Exception as e:
                status, err = "error", str(e)

        rows.append({
            "local_path": str(dst),
            "status": status,
            "error": err,
            "image_url": img_url,
            "observation_url": row.get("url"),
            "license": license_code,
            "taxon_id": taxon_id,
            "scientific_name": sci_name,
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "downloaded_metadata.csv", index=False)

if __name__ == "__main__":
    main()
