# Training

Labeln, Dataset erstellen und YOLO-Modell trainieren.

## Ordnerstruktur

```
train/
├── data/
│   ├── images/              # Quellbilder (Git LFS) — einzige Kopie der Bilder
│   └── labels/
│       ├── classes.txt
│       ├── notes.json
│       └── labels/          # YOLO-Annotationen (.txt), exportiert von Label Studio
├── labelstudio/             # Label Studio DB + Config (kein media/ im Repo)
│   └── label_studio.sqlite3
├── tools/                   # Skripte zur manuellen Inspektion von Kamera und Modell
├── documentation/
├── dataset/                 # generiert von create_dataset.py (nicht im Repo)
├── export_labels.py
├── create_dataset.py
├── augment_dataset.py
├── active_learning.py
├── train.py
├── docker-compose.yml       # startet Label Studio, mountet data/images/
└── requirements.txt
```

## Pipeline

```
Label Studio (manuelles Labeln)
        |
        v
export_labels.py  -->  data/labels/labels/
        |
        v
create_dataset.py  -->  dataset/  (train 70% / val 20% / test 10%)
        |
        v
augment_dataset.py  -->  dataset/train/
        |
        v
train.py  -->  runs/best.pt
        |
        v
active_learning.py  -->  Label Studio (Vorannotierung neuer Frames)
        |
        +-- nach manuellem Review: zurück zu export_labels.py
```

1. **Labeln** — In Label Studio Bilder manuell labeln.
2. **Labels exportieren** — `export_labels.py` holt die Annotationen per API und speichert sie als YOLO-`.txt`-Dateien in `data/labels/labels/`. Danach `data/labels/` committen.
3. **Dataset aufbereiten** — `create_dataset.py` baut ein YOLO-kompatibles Dataset mit Train/Val/Test-Split.
4. **Augmentieren** — `augment_dataset.py` erweitert ausschliesslich die Trainingsdaten.
5. **Trainieren** — `train.py` fine-tuned YOLO26n. Bestes Modell landet unter `runs/`.
6. **Active Learning** — `active_learning.py` erfasst neue Frames von der Kamera und pusht sie mit Vorannotierungen nach Label Studio.

## Einrichtung

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env ausfüllen
```

## Neues Projekt aufsetzen (Ersteinrichtung)

```sh
git clone <repo>          # Git LFS lädt Bilder + SQLite-DB automatisch herunter
cd train
cp .env.example .env      # LABEL_STUDIO_REFRESH_TOKEN + TAPO_* eintragen
docker compose up -d      # Label Studio startet mit allen Bildern und Annotationen
python create_dataset.py  # dataset/ für Training erstellen
```

## Label Studio

```sh
docker compose up -d
```

Label Studio startet auf `http://localhost:8080`. Die Bilder aus `data/images/` sind direkt sichtbar — kein manueller Import nötig.

Login: `bierli01@example.com` / `bierli01@example.com`

## Skripte

#### `export_labels.py`
Exportiert die aktuellen Annotationen aus Label Studio nach `data/labels/labels/`. Nach jeder Labeling-Session ausführen und `data/labels/` committen.
```sh
python export_labels.py
```

#### `create_dataset.py`
Liest Bilder aus `data/images/` und Labels aus `data/labels/labels/`, erstellt `dataset/` mit Train/Val/Test-Split.
```sh
python create_dataset.py
```

#### `augment_dataset.py`
Muss nach `create_dataset.py` und vor `train.py` ausgeführt werden.
```sh
python augment_dataset.py
```

| Parameter | Wert | Beschreibung |
|---|---|---|
| `DATASET_MULTIPLIER` | `1.25` | Gesamtgrösse relativ zum Original |

#### `train.py`
Erkennt automatisch CUDA, Apple MPS oder CPU.
```sh
python train.py
```

#### `active_learning.py`
Erfasst alle 5 Minuten einen Frame, führt Inferenz durch und pusht Vorannotierungen nach Label Studio.
```sh
python active_learning.py
```

#### `tools/tapo_stream.py`
Öffnet den RTSP-Stream der Kamera und zeigt ihn live an. Nützlich zum Überprüfen der Kameraverbindung.
```sh
python tools/tapo_stream.py
```

#### `tools/bee_counter.py`
Zählt einmal pro Minute die erkannten Bienen und speichert das annotierte Bild unter `tools/detections/`.
```sh
python tools/bee_counter.py
```

#### `tools/bee_tracker.py`
Zeigt den RTSP-Stream mit Live-Tracking (IDs) an. Nützlich um die Modellqualität visuell zu beurteilen.
```sh
python tools/bee_tracker.py
```

## Umgebungsvariablen

`.env.example` nach `.env` kopieren:

| Variable | Beschreibung |
|---|---|
| `TAPO_USER` | Benutzername der Tapo-Kamera |
| `TAPO_PASS` | Passwort der Tapo-Kamera |
| `TAPO_HOST` | IP-Adresse der Tapo-Kamera |
| `TAPO_PORT` | RTSP-Port (Standard: 554) |
| `LABEL_STUDIO_URL` | URL der Label-Studio-Instanz |
| `LABEL_STUDIO_PROJECT_ID` | Projekt-ID in Label Studio |
| `LABEL_STUDIO_REFRESH_TOKEN` | JWT Refresh-Token (Account & Settings → Access Token) |
