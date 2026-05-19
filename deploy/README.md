# Deployment

Live-Erkennung und Dashboard laufen als Docker-Container auf einem Server. Der Counter verbindet sich via Tailscale VPN zur Kamera im Heimnetz.

## Übersicht

```
Tapo Kamera (Heimnetz)
        |  RTSP via Tailscale VPN
        v
counter/     — YOLO Inferenz, speichert in SQLite
        |  shared volume (data/)
dashboard/   — FastAPI + Web-Dashboard
        |
https://swarm-alarm.crstn.ch
```

## Ordnerstruktur

```
deploy/
├── counter/          # RTSP → YOLO → SQLite
├── dashboard/        # FastAPI + Dashboard
│   └── static/       # CSS, JS, Favicon
├── model/            # best.pt (via Git LFS)
├── data/             # SQLite DB + Snapshots (nicht im Repo)
├── docker-compose.yml
└── .env.example
```

## Setup

```sh
cd deploy
cp .env.example .env
# .env ausfüllen
docker compose up -d --build
```

## Modell aktualisieren

Trainiertes Modell aus `train/runs/` nach `deploy/model/best.pt` kopieren und neu deployen:

```sh
cp runs/detect/local_bee_models/<run>/weights/best.pt deploy/model/best.pt
cd deploy && docker compose up -d --build
```

## Umgebungsvariablen

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
