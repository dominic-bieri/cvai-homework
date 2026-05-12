# cvai-homework

Bienen-Zählsystem mit YOLO-Objekterkennung, einer Tapo-IP-Kamera (RTSP) und einem Active-Learning-Loop über Label Studio.

## Pipeline

```
Label Studio (manuelles Labeln)
        │
        ▼
create_dataset.py  ──►  self-labled-dataset-yolo/
        │
        ▼
train.py  ──►  runs/ (best.pt)
        │
        ├──►  active_learning.py  ──►  Label Studio (Vorannotierung mit best.pt)
        │              │                      │
        │              └──── mehr Daten ──────┘  (→ nochmals trainieren (train.py))
        │
        └──►  bee_counter.py (Live-Erkennung)
```

1. **Labeln** — In Label Studio Bilder manuell labeln und als YOLO-Export speichern.
2. **Dataset aufbereiten** — `create_dataset.py` baut ein YOLO-kompatibles Dataset (70 % Train / 20 % Val / 10 % Test).
3. **Trainieren** — `train.py` fine-tuned YOLO auf dem Dataset. Bestes Modell landet unter `runs/`.
4. **Active Learning** — `active_learning.py` nutzt das trainierte Modell, um neue Frames automatisch vorannotiert in Label Studio hochzuladen. Nach manuellem Review kann erneut trainiert werden.
5. **Live-Erkennung** — `bee_counter.py` zählt Bienen in Echtzeit und speichert annotierte Bilder unter `detections/`.

---

## Skripte

### `create_dataset.py`
Liest Label-Studio-Exporte und Labels aus `self-labled-dataset/`, mischt sie und teilt sie in Train/Val/Test auf.
```sh
python create_dataset.py
```

### `train.py`
Fine-tuned YOLO (`yolo26n.pt`) auf dem aufbereiteten Dataset. Erkennt automatisch CUDA, Apple MPS oder CPU.
```sh
python train.py
```

### `active_learning.py`
Erfasst alle 5 Minuten einen Frame vom RTSP-Stream, führt Inferenz mit dem trainierten Modell durch und pusht das Bild mit Vorannotierungen nach Label Studio.
```sh
python active_learning.py
```

### `bee_counter.py`
Läuft dauerhaft, fragt den RTSP-Stream alle 60 Sekunden ab, zählt erkannte Bienen und speichert annotierte Bilder im Ordner `detections/`.
```sh
python bee_counter.py
```

---

## Installation

Mac / Linux:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```sh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Umgebungsvariablen

`.env.example` nach `.env` kopieren und ausfüllen:

```sh
cp .env.example .env
```

| Variable | Beschreibung |
|---|---|
| `TAPO_USER` | Benutzername der Tapo-Kamera *(privat)* |
| `TAPO_PASS` | Passwort der Tapo-Kamera *(privat)* |
| `TAPO_HOST` | IP-Adresse der Tapo-Kamera *(privat)* |
| `TAPO_PORT` | RTSP-Port (Standard: 554) |
| `LABEL_STUDIO_URL` | URL der Label-Studio-Instanz |
| `LABEL_STUDIO_PROJECT_ID` | Projekt-ID in Label Studio |
| `LABEL_STUDIO_REFRESH_TOKEN` | JWT Refresh-Token (Account & Settings → Access Token) |

## Label Studio

```sh
docker-compose up -d
```

Login: bierli01@example.com / bierli01@example.com
