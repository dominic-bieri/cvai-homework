# cvai-homework

Bienen-Zählsystem mit YOLO-Objekterkennung, einer Tapo-IP-Kamera (RTSP) und einem Active-Learning-Loop über Label Studio.

## Übersicht

```
Tapo Kamera (Heimnetz)
        |  RTSP
        v
train/active_learning.py  —  erfasst Frames, Vorannotierung via YOLO
        |
        v
Label Studio  —  manuelles Labeln / Korrigieren
        |
        v
train/  —  Export, Dataset erstellen, Trainieren
        |
        v
deploy/  —  Live-Erkennung + Dashboard auf dem Server
```

## Teile

- **[Training](train/README.md)** — Label Studio, Dataset-Pipeline, Modell trainieren
- **[Deployment](deploy/README.md)** — Live-Erkennung, Dashboard, Docker-Setup

## Voraussetzungen

Dieses Repository verwendet [Git LFS](https://git-lfs.com/) für Bilder und Modelle.

```sh
git lfs install
git clone <repo>
```
