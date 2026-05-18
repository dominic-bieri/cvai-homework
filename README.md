# cvai-homework

Bienen-Zählsystem mit YOLO-Objekterkennung, einer Tapo-IP-Kamera (RTSP) und einem Active-Learning-Loop über Label Studio.

---

## Übersicht

Das Projekt besteht aus zwei Teilen:

- **Training** — Labeln, Augmentieren, Trainieren (lokal)
- **Deploy** — Live-Erkennung und Dashboard (Server via Docker)

---

## Training Pipeline

```
Label Studio (manuelles Labeln)
        |
        v
create_dataset.py  -->  self-labled-dataset-yolo/
        |
        v
augment_dataset.py  -->  self-labled-dataset-yolo/train/
        |
        v
train.py  -->  runs/best.pt
        |
        v
active_learning.py  -->  Label Studio (Vorannotierung)
        |
        +-- nach manuellem Review: zurück zu create_dataset.py
```

1. **Labeln** — In Label Studio Bilder manuell labeln und als YOLO-Export speichern.
2. **Dataset aufbereiten** — `create_dataset.py` baut ein YOLO-kompatibles Dataset (70 % Train / 20 % Val / 10 % Test).
3. **Augmentieren** — `augment_dataset.py` erweitert ausschliesslich die Trainingsdaten mit augmentierten Kopien inkl. korrekter Label-Transformation.
4. **Trainieren** — `train.py` fine-tuned YOLO26n auf dem Dataset. Bestes Modell landet unter `runs/`.
5. **Active Learning** — `active_learning.py` nutzt das trainierte Modell, um neue Frames automatisch vorannotiert in Label Studio hochzuladen.

### Skripte

#### `create_dataset.py`
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
| `DATASET_MULTIPLIER` | `4.0` | Gesamtgrösse relativ zum Original |

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

### Installation (Training)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Label Studio (lokal)

```sh
docker compose up -d
```

Login: `bierli01@example.com` / `bierli01@example.com`

### Umgebungsvariablen (Training)

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

---

## Deploy

Live-Erkennung und Dashboard laufen als Docker-Container auf einem Server. Der Counter verbindet sich via Tailscale VPN zur Kamera im Heimnetz.

```
Tapo Kamera (Heimnetz)
        |  RTSP via Tailscale VPN
        v
deploy/counter     — YOLO Inferenz, speichert in SQLite
        |  shared volume (data/)
deploy/dashboard   — FastAPI + Web-Dashboard
        |
https://swarm-alarm.crstn.ch
```

### Struktur

```
deploy/
├── counter/          # RTSP → YOLO → SQLite
├── dashboard/        # FastAPI + Dashboard
│   └── static/       # CSS, JS, Favicon
├── tailscale/        # VPN-Container für Kamerazugriff
├── model/            # best.pt (via Git LFS)
├── data/             # SQLite DB + Snapshots (nicht im Repo)
├── docker-compose.yml
└── .env.example
```

### Setup

```sh
cd deploy
cp .env.example .env
# .env ausfüllen
docker compose up -d --build
```

### Umgebungsvariablen (Deploy)

| Variable | Beschreibung |
|---|---|
| `TAPO_USER` | Benutzername der Tapo-Kamera |
| `TAPO_PASS` | Passwort der Tapo-Kamera |
| `TAPO_HOST` | Tailscale IP der Kamera |
| `TAPO_PORT` | RTSP-Port (Standard: 554) |
| `TS_AUTHKEY` | Tailscale Auth Key |
| `MODEL_PATH` | Pfad zum Modell (Standard: `/model/best.pt`) |
| `INTERVAL_SECONDS` | Messintervall in Sekunden (Standard: 60) |
| `SWARM_THRESHOLD` | Schwarmalarm ab dieser Anzahl Bienen (Standard: 50) |
| `ALARM_COOLDOWN_MINUTES` | Mindestabstand zwischen Alarmen (Standard: 30) |
| `ALARM_RETENTION_DAYS` | Alarme älter als X Tage löschen (Standard: 30) |

---

## Voraussetzungen

Dieses Repository verwendet [Git LFS](https://git-lfs.com/) für grosse Dateien (Bilder, Modelle).

```sh
git lfs install
git clone ...
```