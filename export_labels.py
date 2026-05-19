import io
import os
import json
import base64
import time
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))
LS_REFRESH_TOKEN = os.getenv("LABEL_STUDIO_REFRESH_TOKEN")

OUT = Path("data/labels")

_token_cache: dict = {"access": None, "exp": 0.0}


def _headers() -> dict:
    if not _token_cache["access"] or time.time() > _token_cache["exp"] - 60:
        if not LS_REFRESH_TOKEN:
            raise RuntimeError(
                "LABEL_STUDIO_REFRESH_TOKEN is not set.\n"
                "Add it to your .env file (Label Studio → Account & Settings → Access Token)."
            )
        resp = requests.post(
            f"{LS_URL}/api/token/refresh",
            json={"refresh": LS_REFRESH_TOKEN},
            timeout=10,
        )
        resp.raise_for_status()
        access = resp.json()["access"]
        padding = "=" * (4 - len(access.split(".")[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(access.split(".")[1] + padding))
        _token_cache["access"] = access
        _token_cache["exp"] = float(payload["exp"])

    return {"Authorization": f"Bearer {_token_cache['access']}"}


def export_labels() -> None:
    print(f"Exporting YOLO labels from project {LS_PROJECT_ID} at {LS_URL} ...")
    resp = requests.get(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/export",
        params={"exportType": "YOLO"},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()

    labels_dir = OUT / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.startswith("labels/") and name.endswith(".txt"):
                (labels_dir / Path(name).name).write_bytes(z.read(name))
            elif name in {"classes.txt", "notes.json"}:
                (OUT / name).write_bytes(z.read(name))

    label_count = sum(1 for _ in labels_dir.glob("*.txt"))
    print(f"Done — {label_count} label files written to {OUT}/")


if __name__ == "__main__":
    export_labels()
