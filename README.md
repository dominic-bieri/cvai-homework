# cvai-homework

Bienen-Zählsystem mit YOLO-Objekterkennung, einer Tapo-IP-Kamera (RTSP) und einem Active-Learning-Loop über Label Studio.

## Pipeline-Übersicht

```
Tapo-Kamera (RTSP)
       │
       ▼
active_learning.py  ──►  Label Studio (Vorannotierung)
                                │
                         manuelles Review
                                │
                         self-labled-dataset/
                                │
                                ▼
                       create_dataset.py  ──►  self-labled-dataset-yolo/
                                                        │
                                                        ▼
                                               train.py  ──►  runs/ (best.pt)
                                                        │
                                                        ▼
                                               bee_counter.py (Live-Erkennung)
```

1. **Daten sammeln** — `active_learning.py` nimmt alle 5 Minuten ein Bild vom RTSP-Stream, führt YOLO-Inferenz durch und lädt das Bild mit Vorvorannotierungen nach Label Studio hoch.
2. **Labeln** — In Label Studio werden die Vorvorannotierungen manuell korrigiert und bestätigt.
3. **Dataset aufbereiten** — `create_dataset.py` baut aus den Label-Studio-Exporten ein YOLO-kompatibles Dataset (70 % Train / 20 % Val / 10 % Test).
4. **Trainieren** — `train.py` fine-tuned YOLO auf dem erstellten Dataset. Das beste Modell landet unter `runs/`.
5. **Live-Erkennung** — `bee_counter.py` zählt Bienen in Echtzeit und speichert annotierte Bilder im Ordner `detections/`.

---

## Skripte

### `tapo_stream.py`
Zeigt den RTSP-Live-Stream der Tapo-Kamera in einem OpenCV-Fenster an. Nützlich zum Prüfen, ob die Kamera erreichbar ist.
```sh
python tapo_stream.py
```
> **Hinweis:** Der Stream funktioniert nur mit der richtigen Tailscale-Konfiguration, da sich die Kamera in einem privaten Netzwerk befindet. IP-Adresse, Benutzername und Passwort sind nicht öffentlich.

### `active_learning.py`
Erfasst über einen konfigurierbaren Zeitraum (Standard: 5 Stunden) alle 5 Minuten einen Frame, führt YOLO-Inferenz durch und pusht das Bild mit Bounding-Box-Vorvorannotierungen in das Label-Studio-Projekt. Kann auch als Modul importiert werden:
```python
from active_learning import push_frame_to_label_studio
push_frame_to_label_studio(frame, model)
```
Standalone-Start (erfasst einen einzelnen Frame):
```sh
python active_learning.py
```

### `create_dataset.py`
Liest die von Label Studio exportierten Bilder (`labelstudio-data/media/upload/1/`) und die zugehörigen YOLO-Labels (`self-labled-dataset/labels/`), mischt sie zufällig und teilt sie in Train/Val/Test auf. Das fertige Dataset wird nach `self-labled-dataset-yolo/` geschrieben.
```sh
python create_dataset.py
```

### `train.py`
Fine-tuned das YOLO-Modell (`yolo26n.pt`) auf dem aufbereiteten Dataset. Erkennt automatisch CUDA, Apple MPS oder CPU. Das beste Modell wird unter `runs/detect/local_bee_models/yolo26_run_01/weights/best.pt` gespeichert.
```sh
python train.py
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

Starten:
```sh
docker-compose up -d
```

Login: bierli01@example.com / bierli01@example.com

