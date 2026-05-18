import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiosqlite
from pathlib import Path
from datetime import datetime, timedelta, timezone

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_PATH = Path("/data/bees.db")
SNAPSHOTS_DIR = Path("/data/snapshots")
SWARM_THRESHOLD = int(os.getenv("SWARM_THRESHOLD", "50"))
ALARM_COOLDOWN_MINUTES = int(os.getenv("ALARM_COOLDOWN_MINUTES", "30"))
ALARM_RETENTION_DAYS = int(os.getenv("ALARM_RETENTION_DAYS", "30"))
DASHBOARD_REFRESH_SECONDS = int(os.getenv("INTERVAL_SECONDS", "60"))

PERIOD_CONFIG = {
    "1h": (
        timedelta(hours=1),
        "strftime('%Y-%m-%dT%H:%MZ', timestamp)",
        "strftime('%Y-%m-%dT%H:%M', timestamp)",
    ),
    "24h": (
        timedelta(hours=24),
        "strftime('%Y-%m-%dT%H:00Z', timestamp)",
        "strftime('%Y-%m-%dT%H', timestamp)",
    ),
    "30d": (timedelta(days=30), "DATE(timestamp)", "DATE(timestamp)"),
    "365d": (timedelta(days=365), "DATE(timestamp)", "DATE(timestamp)"),
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_since(delta: timedelta) -> str:
    return (now_utc() - delta).strftime("%Y-%m-%dT%H:%M:%S")


async def query(sql: str, params: tuple = ()) -> list[dict]:
    if not DB_PATH.exists():
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            return [dict(r) for r in await cursor.fetchall()]


async def execute(sql: str, params: tuple = ()) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(sql, params)
        await db.commit()


@app.on_event("startup")
async def startup():
    if not DB_PATH.exists():
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS swarm_alarms (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                count     INTEGER NOT NULL,
                threshold INTEGER NOT NULL
            )
        """)
        await db.commit()
    # Clean up old alarms on startup
    await execute(
        "DELETE FROM swarm_alarms WHERE timestamp < ?",
        (utc_since(timedelta(days=ALARM_RETENTION_DAYS)),),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/config")
async def api_config():
    return {"threshold": SWARM_THRESHOLD, "refresh_seconds": DASHBOARD_REFRESH_SECONDS}


@app.get("/api/aggregated")
async def api_aggregated(period: str = "24h"):
    if period not in PERIOD_CONFIG:
        raise HTTPException(400, "Invalid period. Use: 1h, 24h, 30d, 365d")
    delta, label_expr, group_expr = PERIOD_CONFIG[period]
    return await query(
        f"""
        SELECT {label_expr}         AS label,
               MAX(count)           AS count,
               ROUND(AVG(count), 1) AS avg
        FROM counts
        WHERE timestamp >= ?
        GROUP BY {group_expr}
        ORDER BY timestamp ASC
    """,
        (utc_since(delta),),
    )


@app.get("/api/stats")
async def api_stats():
    today = now_utc().strftime("%Y-%m-%dT00:00:00")
    latest = await query("SELECT count, timestamp FROM counts ORDER BY id DESC LIMIT 1")
    peak = await query(
        "SELECT MAX(count) as peak FROM counts WHERE timestamp >= ?", (today,)
    )
    avg = await query(
        "SELECT AVG(count)  as avg  FROM counts WHERE timestamp >= ?", (today,)
    )
    total = await query("SELECT COUNT(*)    as total FROM counts")

    hourly_raw = await query("""
        SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
               ROUND(AVG(count), 1)                       as avg_count
        FROM counts GROUP BY hour ORDER BY hour ASC
    """)
    hourly_map = {r["hour"]: r["avg_count"] for r in hourly_raw}

    return {
        "latest": latest[0] if latest else {"count": 0, "timestamp": None},
        "peak": int(peak[0]["peak"] or 0),
        "avg": round(float(avg[0]["avg"] or 0), 1),
        "total_observations": int(total[0]["total"] or 0),
        "hourly": [{"hour": h, "avg_count": hourly_map.get(h, 0)} for h in range(24)],
    }


@app.get("/api/alarms")
async def api_alarms():
    return await query(
        "SELECT timestamp, count, threshold FROM swarm_alarms ORDER BY id DESC LIMIT 100"
    )


class AlarmCheck(BaseModel):
    threshold: int


@app.post("/api/alarms/check")
async def check_alarms(body: AlarmCheck):
    latest = await query("SELECT count, timestamp FROM counts ORDER BY id DESC LIMIT 1")
    if not latest:
        return {"alarm": False}

    row = latest[0]
    if row["count"] >= body.threshold:
        recent = await query(
            "SELECT id FROM swarm_alarms WHERE timestamp >= ?",
            (utc_since(timedelta(minutes=ALARM_COOLDOWN_MINUTES)),),
        )
        if not recent:
            ts = utc_str()
            await execute(
                "INSERT INTO swarm_alarms (timestamp, count, threshold) VALUES (?, ?, ?)",
                (ts, row["count"], body.threshold),
            )
            return {"alarm": True, "count": row["count"], "timestamp": ts}

    return {"alarm": False}


@app.get("/api/snapshot")
async def api_snapshot():
    if not SNAPSHOTS_DIR.exists():
        return {"url": None}
    snapshots = sorted(SNAPSHOTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        return {"url": None}
    latest = snapshots[-1]
    return {
        "url": f"/snapshot/{latest.name}",
        "timestamp": latest.stem.replace("bee_", ""),
    }


@app.get("/snapshot/{filename}")
async def serve_snapshot(filename: str):
    path = (SNAPSHOTS_DIR / filename).resolve()
    if not str(path).startswith(str(SNAPSHOTS_DIR.resolve())):
        raise HTTPException(400)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")
