#!/usr/bin/env python3
"""
music-api — Lossless music search & download server.

Runs on the home server alongside slskd + Navidrome.
Exposes REST API for the CLI client and iPhone Shortcuts.
"""
import asyncio
import contextlib
import logging
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine import (
    search_batch, download_track, UnifiedResult, DownloadResult,
    upgrade_scan, upgrade_download, _is_non_original, _normalize_for_match,
    retag_library,
)
from recommender import (
    get_recommendations, create_navidrome_playlist, cleanup_recommendations,
    RECOMMEND_PLAYLIST, PROVIDERS as LLM_PROVIDERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("music-api")

# ─── Config ───────────────────────────────────────────────────────

MUSIC_DIR = os.getenv("MUSIC_DIR", "/music")
SLSKD_URL = os.getenv("SLSKD_URL", "http://slskd:5030")
SLSKD_KEY = os.getenv("SLSKD_KEY", "")
UPGRADE_INTERVAL = int(os.getenv("UPGRADE_INTERVAL", "14400"))  # 4 hours
API_KEY = os.getenv("API_KEY", "")

# AI Recommendations config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")  # openai, deepseek, claude, openrouter
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # override default model per provider
NAVIDROME_URL = os.getenv("NAVIDROME_URL", "http://navidrome:4533")
NAVIDROME_USER = os.getenv("NAVIDROME_USER", "")
NAVIDROME_PASSWORD = os.getenv("NAVIDROME_PASSWORD", "")
RECOMMEND_INTERVAL = int(os.getenv("RECOMMEND_INTERVAL", "0"))  # 0 = disabled
RECOMMEND_CLEANUP_HOURS = int(os.getenv("RECOMMEND_CLEANUP_HOURS", "24"))

# ─── Background upgrader ─────────────────────────────────────────

_upgrader_task: asyncio.Task | None = None
_recommender_task: asyncio.Task | None = None
_cleanup_task: asyncio.Task | None = None
_upgrade_status: dict = {"last_run": None, "last_result": None, "next_run": None}
_recommend_status: dict = {"last_run": None, "last_result": None, "enabled": False}


async def _upgrader_loop():
    """Background loop: periodically scan for 16→24bit upgrades on Soulseek."""
    await asyncio.sleep(300)  # Wait 5 minutes before first scan
    while True:
        _upgrade_status["next_run"] = time.time()
        try:
            logger.info("[UPGRADE] Scan starting...")
            upgrades = await upgrade_scan(MUSIC_DIR, SLSKD_URL, SLSKD_KEY)
            logger.info("[UPGRADE] Found %d candidates", len(upgrades))

            upgraded = 0
            for u in upgrades:
                try:
                    dl = await upgrade_download(u, MUSIC_DIR, SLSKD_URL, SLSKD_KEY)
                    if dl:
                        upgraded += 1
                        logger.info("[UPGRADE] ✅ %s - %s (%dbit → %dbit)",
                                    u["track"]["artist"], u["track"]["title"],
                                    u["track"]["bit_depth"], dl.bit_depth)
                except Exception:
                    logger.exception("[UPGRADE] ❌ Failed: %s", u["track"].get("title"))
                await asyncio.sleep(5)

            logger.info("[UPGRADE] Done: %d/%d upgraded", upgraded, len(upgrades))
            _upgrade_status["last_run"] = time.time()
            _upgrade_status["last_result"] = {
                "candidates": len(upgrades), "upgraded": upgraded,
                "timestamp": time.time(),
            }
        except Exception:
            logger.exception("[UPGRADE] Scan failed")

        _upgrade_status["next_run"] = time.time() + UPGRADE_INTERVAL
        await asyncio.sleep(UPGRADE_INTERVAL)


async def _recommender_loop():
    """Background loop: periodically generate AI recommendations."""
    await asyncio.sleep(300)  # Wait 5 min after startup
    while True:
        if not (LLM_PROVIDER and LLM_API_KEY and NAVIDROME_USER):
            await asyncio.sleep(3600)
            continue
        try:
            await _run_recommendation_cycle()
        except Exception:
            logger.exception("[RECOMMEND] Auto-cycle failed")
        await asyncio.sleep(RECOMMEND_INTERVAL)


async def _cleanup_loop():
    """Background loop: clean up expired recommendation playlists."""
    await asyncio.sleep(600)
    while True:
        if not (NAVIDROME_USER and NAVIDROME_PASSWORD):
            await asyncio.sleep(3600)
            continue
        try:
            result = await cleanup_recommendations(
                MUSIC_DIR, NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD,
            )
            if result.get("status") == "done":
                logger.info("[CLEANUP] Recommendations: kept %d, deleted %d",
                            result["kept"], result["deleted"])
        except Exception:
            logger.exception("[CLEANUP] Failed")
        await asyncio.sleep(RECOMMEND_CLEANUP_HOURS * 3600)


async def _run_recommendation_cycle():
    """Full recommendation cycle: AI → search → download → playlist."""
    logger.info("[RECOMMEND-AUTO] Generating AI recommendations...")

    recs = await get_recommendations(
        MUSIC_DIR, LLM_PROVIDER, LLM_API_KEY,
        model=LLM_MODEL or None,
    )
    if not recs:
        logger.warning("[RECOMMEND-AUTO] No recommendations from AI")
        return

    logger.info("[RECOMMEND-AUTO] Got %d recs, searching & downloading...", len(recs))

    # Search all tracks
    results = await search_batch(
        recs, slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY,
    )

    # Download to _recommendations subdirectory
    rec_dir = os.path.join(MUSIC_DIR, "_recommendations")
    os.makedirs(rec_dir, exist_ok=True)

    downloaded_paths = []
    for r in results:
        if not r.best:
            continue
        try:
            dl = await download_track(r, rec_dir, SLSKD_URL, SLSKD_KEY)
            if dl and dl.path:
                downloaded_paths.append(dl.path)
        except Exception:
            logger.warning("[RECOMMEND-AUTO] ❌ Not found: %s - %s", r.artist, r.title)

    logger.info("[RECOMMEND-AUTO] Downloaded %d/%d", len(downloaded_paths), len(recs))

    # Create Navidrome playlist
    if downloaded_paths and NAVIDROME_USER and NAVIDROME_PASSWORD:
        playlist_id = await create_navidrome_playlist(
            NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD,
            downloaded_paths, MUSIC_DIR,
        )
        if playlist_id:
            logger.info("[RECOMMEND-AUTO] Playlist created: %s", playlist_id)

    _recommend_status["last_run"] = time.time()
    _recommend_status["last_result"] = {
        "recommended": len(recs),
        "downloaded": len(downloaded_paths),
        "timestamp": time.time(),
    }


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _upgrader_task, _recommender_task, _cleanup_task
    _upgrader_task = asyncio.create_task(_upgrader_loop())
    logger.info("[STARTUP] Background upgrader (interval: %ds)", UPGRADE_INTERVAL)

    if RECOMMEND_INTERVAL > 0 and LLM_PROVIDER and LLM_API_KEY:
        _recommender_task = asyncio.create_task(_recommender_loop())
        _recommend_status["enabled"] = True
        logger.info("[STARTUP] AI recommender (interval: %ds, provider: %s)", RECOMMEND_INTERVAL, LLM_PROVIDER)
    if NAVIDROME_USER and NAVIDROME_PASSWORD:
        _cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("[STARTUP] Recommendation cleanup (every %dh)", RECOMMEND_CLEANUP_HOURS)

    yield
    for task in [_upgrader_task, _recommender_task, _cleanup_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Musomatic API", version="2.0", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Health check always public (but returns minimal info)
    if request.url.path == "/health":
        return await call_next(request)
    # Require API key for all other endpoints
    if API_KEY:
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


# ─── In-memory job tracking ──────────────────────────────────────

_jobs: dict[str, dict] = {}
_cancel_events: dict[str, asyncio.Event] = {}
_MAX_JOBS = 100
_JOB_TTL = 3600

_TERMINAL_STATES = {"done", "failed", "cancelled", "not_found"}


def _cleanup_jobs() -> None:
    now = time.time()
    expired = [
        jid for jid, j in _jobs.items()
        if j["status"] in _TERMINAL_STATES and now - j.get("started", 0) > _JOB_TTL
    ]
    for jid in expired:
        _jobs.pop(jid, None)
        _cancel_events.pop(jid, None)
    if len(_jobs) > _MAX_JOBS:
        terminal = sorted(
            ((jid, j) for jid, j in _jobs.items() if j["status"] in _TERMINAL_STATES),
            key=lambda x: x[1].get("started", 0),
        )
        for jid, _ in terminal[:len(_jobs) - _MAX_JOBS]:
            _jobs.pop(jid, None)
            _cancel_events.pop(jid, None)


# ─── Library caches ──────────────────────────────────────────────

_stats_cache: dict = {"data": None, "ts": 0}
_library_cache: dict = {"data": None, "ts": 0}
_LIBRARY_CACHE_TTL = 30


def _invalidate_caches():
    _stats_cache["data"] = None
    _stats_cache["ts"] = 0
    _library_cache["data"] = None
    _library_cache["ts"] = 0


# ─── Models ───────────────────────────────────────────────────────

class TrackQuery(BaseModel):
    artist: str
    title: str

class SearchRequest(BaseModel):
    tracks: list[TrackQuery]

class DownloadRequest(BaseModel):
    tracks: list[TrackQuery]

class SingleDownload(BaseModel):
    artist: str
    title: str

class DeleteRequest(BaseModel):
    query: str = ""
    ids: list[int] = []

class QuickDownload(BaseModel):
    query: str  # "Artist - Title" or just "Title"


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "tracks": _count_flacs()}


@app.get("/search/browse")
async def search_browse(q: str, limit: int = 30):
    """Browse Tidal catalog — returns matching tracks with quality info."""
    import httpx as hx
    proxy = os.getenv("PROXY_URL") or None
    api_base = os.getenv("MONOCHROME_API", "https://api.monochrome.tf")
    try:
        async with hx.AsyncClient(proxy=proxy, timeout=15) as c:
            r = await c.get(f"{api_base}/search/", params={"s": q, "limit": min(limit, 50)})
            r.raise_for_status()
            items = r.json().get("data", {}).get("items", [])
    except Exception as e:
        raise HTTPException(502, f"Tidal search failed: {e}")

    results = []
    for item in items:
        tags = item.get("mediaMetadata", {}).get("tags", [])
        if "HIRES_LOSSLESS" in tags:
            quality = "Hi-Res"
        elif item.get("audioQuality") == "LOSSLESS" or "LOSSLESS" in tags:
            quality = "Lossless"
        else:
            quality = "Lossy"
        artist = item.get("artist", {}).get("name", "")
        results.append({
            "artist": artist,
            "title": item.get("title", ""),
            "album": item.get("album", {}).get("title", ""),
            "quality": quality,
            "duration_s": item.get("duration", 0),
            "version": item.get("version", "") or "",
        })
    return {"results": results, "total": len(results)}


@app.post("/search")
async def search_tracks(req: SearchRequest):
    t0 = time.time()
    queries = [{"artist": t.artist, "title": t.title} for t in req.tracks]
    progress = {"phase": "", "done": 0, "total": 0}

    def on_progress(done, total, phase):
        progress.update(phase=phase, done=done, total=total)

    results = await search_batch(
        queries, slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY,
        on_progress=on_progress,
    )
    elapsed = time.time() - t0
    hires = sum(1 for r in results if r.best and r.best.is_hires)
    cd = sum(1 for r in results if r.best and r.best.is_lossless and not r.best.is_hires)
    not_found = sum(1 for r in results if not r.best)

    return {
        "results": [r.to_dict() for r in results],
        "stats": {
            "total": len(results), "hires": hires, "cd": cd,
            "not_found": not_found,
            "lossless_pct": round((hires + cd) / max(len(results), 1) * 100),
            "elapsed_s": round(elapsed, 1),
        },
    }


@app.post("/download")
async def download_single(req: SingleDownload, bg: BackgroundTasks):
    _cleanup_jobs()
    job_id = f"dl-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "searching", "artist": req.artist, "title": req.title,
        "result": None, "error": None, "started": time.time(),
    }
    _cancel_events[job_id] = asyncio.Event()
    bg.add_task(_download_job, job_id, req.artist, req.title)
    return {"job_id": job_id, "status": "searching"}


def _parse_query(query: str) -> tuple[str, str]:
    for sep in [" - ", " — ", " – "]:
        if sep in query:
            artist, title = query.split(sep, 1)
            return artist.strip(), title.strip()
    return "", query.strip()


@app.post("/quick")
async def quick_download(req: QuickDownload, bg: BackgroundTasks):
    """Async search + download. Returns instantly.
    If track already exists — returns immediately with status 'exists'.
    Otherwise starts background download, returns job_id.
    Use GET /jobs/{job_id} to poll progress.
    ?wait=1 for old sync behavior (CLI usage)."""
    from engine import _find_existing
    artist, title = _parse_query(req.query)
    if not title:
        raise HTTPException(400, "Empty query")

    logger.info("[QUICK-DL] %s - %s", artist or "?", title)
    t0 = time.time()

    # Fast dedup check — instant response for existing tracks
    existing = _find_existing(MUSIC_DIR, artist or "", title)
    if existing:
        from mutagen.flac import FLAC as FLACInfo
        bd, sr = 16, 44100
        try:
            info = FLACInfo(existing).info
            bd = info.bits_per_sample or 16
            sr = info.sample_rate or 44100
        except Exception:
            pass
        elapsed = round(time.time() - t0, 1)
        rate = f"{sr / 1000:.1f}kHz"
        quality = f"Hi-Res {bd}bit/{rate}" if bd >= 24 else f"CD {bd}bit/{rate}"
        return {
            "status": "exists",
            "message": f"♻️ {artist or '?'} — {title}\n🎵 {quality}",
            "artist": artist, "title": title, "album": "",
            "quality": quality, "elapsed_s": elapsed,
        }

    # Start async download — return immediately
    _cleanup_jobs()
    job_id = f"quick-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "searching", "artist": artist, "title": title,
        "result": None, "error": None, "started": time.time(),
    }
    _cancel_events[job_id] = asyncio.Event()
    bg.add_task(_quick_download_job, job_id, artist, title)
    return {
        "status": "downloading",
        "message": f"⬇️ {artist + ' — ' if artist else ''}{title}",
        "job_id": job_id,
        "artist": artist, "title": title,
    }


async def _quick_download_job(job_id: str, artist: str, title: str):
    """Background job for /quick endpoint."""
    try:
        results = await search_batch(
            [{"artist": artist, "title": title}],
            slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY,
        )
        r = results[0]
        if not r.best:
            _jobs[job_id]["status"] = "not_found"
            _jobs[job_id]["error"] = f"Not found: {artist} — {title}" if artist else f"Not found: {title}"
            return

        _jobs[job_id]["status"] = "downloading"
        dl = await download_track(r, MUSIC_DIR, SLSKD_URL, SLSKD_KEY)

        if dl:
            _invalidate_caches()
            bd = dl.bit_depth
            sr = dl.sample_rate
            rate = f"{sr / 1000:.1f}kHz"
            quality = f"Hi-Res {bd}bit/{rate}" if bd >= 24 else f"CD {bd}bit/{rate}"
            _jobs[job_id].update({
                "status": "done", "result": dl,
                "artist": dl.artist, "title": dl.title,
                "album": getattr(dl, 'album', ''),
                "quality": quality,
            })
        else:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "All sources failed"
    except Exception as e:
        logger.exception("[QUICK-JOB] %s", e)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ─── Library Management ──────────────────────────────────────────

def _scan_library() -> list[dict]:
    now = time.time()
    if _library_cache["data"] is not None and now - _library_cache["ts"] < _LIBRARY_CACHE_TTL:
        return _library_cache["data"]

    from mutagen.flac import FLAC
    music = Path(MUSIC_DIR)
    tracks = []
    for i, f in enumerate(sorted(music.rglob("*.flac")), 1):
        try:
            audio = FLAC(str(f))
            artist = audio.get("artist", [""])[0]
            title = audio.get("title", [""])[0]
            album = audio.get("album", [""])[0]
        except Exception:
            artist, title, album = "", "", ""
        tracks.append({
            "id": i,
            "path": str(f.relative_to(music)),
            "artist": artist or f.parent.name,
            "title": title or f.stem,
            "album": album,
            "size_mb": round(f.stat().st_size / 1_000_000, 1),
            "bit_depth": audio.info.bits_per_sample if audio and audio.info else 0,
            "sample_rate": audio.info.sample_rate if audio and audio.info else 0,
        })
    _library_cache["data"] = tracks
    _library_cache["ts"] = now
    return tracks


@app.get("/library/tracks")
def library_tracks(q: str = ""):
    tracks = _scan_library()
    if q:
        q_lower = q.lower()
        tracks = [t for t in tracks if q_lower in f"{t['artist']} {t['title']} {t['album']}".lower()]
    return {"tracks": tracks, "total": len(tracks)}


@app.post("/library/delete")
def library_delete(req: DeleteRequest):
    all_tracks = _scan_library()
    if req.ids:
        id_set = set(req.ids)
        matches = [t for t in all_tracks if t["id"] in id_set]
    elif req.query and len(req.query) >= 2:
        q_lower = req.query.lower()
        matches = [t for t in all_tracks if q_lower in f"{t['artist']} {t['title']} {t['album']}".lower()]
    else:
        raise HTTPException(400, "Provide ids or query (min 2 chars)")

    if not matches:
        return {"status": "not_found", "message": "❌ No matches", "deleted": 0}

    deleted = []
    music = Path(MUSIC_DIR).resolve()
    for t in matches:
        fp = (music / t["path"]).resolve()
        if not str(fp).startswith(str(music)):
            logger.warning("Path traversal blocked: %s", t["path"])
            continue
        try:
            fp.unlink()
            parent = fp.parent
            if parent != music and not any(parent.iterdir()):
                parent.rmdir()
            deleted.append(f"{t['artist']} — {t['title']}")
        except Exception as e:
            logger.warning("Failed to delete %s: %s", t["path"], e)

    _invalidate_caches()
    names = "\n".join(deleted)
    return {
        "status": "done",
        "message": f"🗑 Deleted ({len(deleted)}):\n{names}",
        "deleted": len(deleted), "tracks": deleted,
    }


@app.get("/library/audit")
async def library_audit():
    from mutagen.flac import FLAC
    music = Path(MUSIC_DIR)
    issues = []
    quality_stats = {"hires": 0, "cd": 0, "other": 0}

    for f in sorted(music.rglob("*.flac")):
        try:
            audio = FLAC(str(f))
            bd = audio.info.bits_per_sample
            artist = (audio.get("artist") or [""])[0]
            title = (audio.get("title") or [""])[0]
            album = (audio.get("album") or [""])[0]
        except Exception:
            continue

        if bd >= 24:
            quality_stats["hires"] += 1
        elif bd == 16:
            quality_stats["cd"] += 1
        else:
            quality_stats["other"] += 1

        fname = f.stem
        if _is_non_original("", title, result_album=album) or _is_non_original("", fname):
            issues.append({
                "type": "non_original",
                "path": str(f.relative_to(music)),
                "artist": artist, "title": title, "album": album,
                "reason": "Remix/live/edit/cover markers detected",
            })

    return {
        "total_tracks": sum(quality_stats.values()),
        "quality": quality_stats,
        "issues": issues, "issue_count": len(issues),
    }


@app.post("/batch/download")
async def batch_download(req: DownloadRequest, bg: BackgroundTasks):
    if len(req.tracks) > 1000:
        raise HTTPException(400, "Batch too large (max 1000)")
    _cleanup_jobs()
    job_id = f"batch-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "scanning", "total": len(req.tracks), "done": 0,
        "hires": 0, "cd": 0, "not_found": 0, "downloaded": 0,
        "failed": 0, "results": [], "started": time.time(),
    }
    _cancel_events[job_id] = asyncio.Event()
    bg.add_task(_batch_job, job_id, req.tracks)
    return {"job_id": job_id, "status": "scanning", "total": len(req.tracks)}


# ─── Upgrade Endpoints ───────────────────────────────────────────

@app.post("/upgrade/trigger")
async def trigger_upgrade(bg: BackgroundTasks):
    _cleanup_jobs()
    job_id = f"upgrade-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "scanning", "started": time.time(),
        "candidates": 0, "upgraded": 0, "failed": 0,
    }
    bg.add_task(_upgrade_job, job_id)
    return {"job_id": job_id, "status": "scanning"}


@app.get("/upgrade/status")
def upgrade_status():
    now = time.time()
    next_run = _upgrade_status.get("next_run")
    return {
        "interval_s": UPGRADE_INTERVAL,
        "last_run": _upgrade_status.get("last_run"),
        "last_result": _upgrade_status.get("last_result"),
        "next_run": next_run,
        "next_run_in_s": round(next_run - now, 1) if next_run and next_run > now else None,
    }


# ─── AI Recommendation Endpoints ─────────────────────────────────

class RecommendRequest(BaseModel):
    provider: str = ""
    model: str = ""
    count: int = 30


@app.post("/recommend/generate")
async def generate_recommendations(req: RecommendRequest, bg: BackgroundTasks):
    """Generate AI recommendations, download them, create Navidrome playlist."""
    provider = req.provider or LLM_PROVIDER
    api_key = LLM_API_KEY
    if not provider or not api_key:
        raise HTTPException(400, "LLM provider and API key required (set LLM_PROVIDER + LLM_API_KEY env vars)")
    count = min(req.count, 50)

    _cleanup_jobs()
    job_id = f"recommend-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "generating", "started": time.time(),
        "recommended": 0, "downloaded": 0, "playlist_id": None,
    }
    bg.add_task(_recommend_job, job_id, provider, api_key, req.model or LLM_MODEL, count)
    return {"job_id": job_id, "status": "generating"}


@app.get("/recommend/status")
def recommend_status():
    return {
        "enabled": _recommend_status.get("enabled", False),
        "provider": LLM_PROVIDER or None,
        "interval_s": RECOMMEND_INTERVAL,
        "cleanup_hours": RECOMMEND_CLEANUP_HOURS,
        "last_run": _recommend_status.get("last_run"),
        "last_result": _recommend_status.get("last_result"),
        "supported_providers": list(LLM_PROVIDERS.keys()),
    }


@app.post("/recommend/cleanup")
async def trigger_cleanup():
    """Manually clean up recommendation playlist. Rated tracks are kept."""
    if not (NAVIDROME_USER and NAVIDROME_PASSWORD):
        raise HTTPException(400, "Navidrome credentials required (NAVIDROME_USER + NAVIDROME_PASSWORD)")
    result = await cleanup_recommendations(
        MUSIC_DIR, NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD,
    )
    return result


# ─── Library Retag ───────────────────────────────────────────────

@app.post("/retag")
async def retag_all(bg: BackgroundTasks):
    """Re-scan library and fill missing metadata from Tidal."""
    _cleanup_jobs()
    job_id = f"retag-{secrets.token_hex(8)}"
    _jobs[job_id] = {
        "status": "running", "started": time.time(),
        "progress": 0, "total": 0, "retagged": 0,
    }

    async def _run_retag():
        try:
            def on_progress(done, total, retagged):
                _jobs[job_id].update(progress=done, total=total, retagged=retagged)
            result = await retag_library(MUSIC_DIR, on_progress=on_progress)
            _jobs[job_id].update(status="done", **result)
            logger.info("[RETAG] Complete: %d retagged, %d failed, %d skipped out of %d",
                        result["retagged"], result["failed"], result["skipped"], result["total"])
        except Exception as e:
            _jobs[job_id].update(status="error", error=str(e))
            logger.error("[RETAG] Failed: %s", e, exc_info=True)

    bg.add_task(_run_retag)
    return {"job_id": job_id, "status": "running"}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("done", "cancelled"):
        return {"status": job["status"], "message": "Already finished"}
    evt = _cancel_events.get(job_id)
    if evt:
        evt.set()
    job["status"] = "cancelling"
    return {"status": "cancelling", "message": "Cancel signal sent"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/jobs")
def list_jobs():
    out = {}
    for jid, j in _jobs.items():
        info = {
            "status": j["status"],
            "started": j.get("started"),
            "artist": j.get("artist", ""),
            "title": j.get("title", ""),
            "error": j.get("error"),
        }
        result = j.get("result")
        if isinstance(result, dict):
            info["quality"] = result.get("quality", "")
            info["album"] = result.get("album", "")
            info["source"] = result.get("stream_type", "")
        out[jid] = info
    return out


@app.get("/library/stats")
def library_stats():
    now = time.time()
    if _stats_cache["data"] is not None and now - _stats_cache["ts"] < 60:
        return _stats_cache["data"]
    music = Path(MUSIC_DIR)
    flacs = list(music.rglob("*.flac"))
    total_size = sum(f.stat().st_size for f in flacs)
    data = {
        "total_tracks": len(flacs),
        "total_size_gb": round(total_size / 1_000_000_000, 2),
        "albums": len(set(f.parent.name for f in flacs)),
    }
    _stats_cache["data"] = data
    _stats_cache["ts"] = now
    return data


# ─── Background Jobs ─────────────────────────────────────────────

async def _download_job(job_id: str, artist: str, title: str):
    job = _jobs[job_id]
    cancel = _cancel_events.get(job_id)
    try:
        if cancel and cancel.is_set():
            job.update(status="cancelled")
            return

        results = await search_batch(
            [{"artist": artist, "title": title}],
            slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY,
        )
        r = results[0]
        if not r.best:
            job.update(status="not_found", error="Track not found on any source")
            return

        if cancel and cancel.is_set():
            job.update(status="cancelled")
            return

        job["status"] = "downloading"
        job["source"] = r.best.to_dict()
        dl = await download_track(r, MUSIC_DIR, SLSKD_URL, SLSKD_KEY)
        if dl:
            _invalidate_caches()
            job.update(status="done", result=dl.to_dict())
        else:
            job.update(status="failed", error="All download sources failed")
    except Exception as e:
        job.update(status="failed", error=str(e))
        logger.exception("[BATCH] Job %s failed", job_id)
    finally:
        _cancel_events.pop(job_id, None)


async def _batch_job(job_id: str, tracks: list[TrackQuery]):
    job = _jobs[job_id]
    cancel = _cancel_events.get(job_id)
    queries = [{"artist": t.artist, "title": t.title} for t in tracks]

    def on_progress(done, total, phase):
        job.update(scan_phase=phase, done=done)

    if cancel and cancel.is_set():
        job.update(status="cancelled")
        _cancel_events.pop(job_id, None)
        return

    results = await search_batch(
        queries, slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY,
        on_progress=on_progress,
    )

    hires = sum(1 for r in results if r.best and r.best.is_hires)
    cd = sum(1 for r in results if r.best and r.best.is_lossless and not r.best.is_hires)
    not_found = sum(1 for r in results if not r.best)
    job.update(status="downloading", hires=hires, cd=cd, not_found=not_found)

    if cancel and cancel.is_set():
        job.update(status="cancelled", elapsed_s=round(time.time() - job["started"], 1))
        _cancel_events.pop(job_id, None)
        return

    downloaded = 0
    failed = 0
    dl_results = []
    for i, r in enumerate(r for r in results if r.best):
        if cancel and cancel.is_set():
            break
        try:
            dl = await download_track(r, MUSIC_DIR, SLSKD_URL, SLSKD_KEY)
            if dl:
                downloaded += 1
                dl_results.append(dl.to_dict())
            else:
                failed += 1
        except Exception:
            failed += 1
            logger.exception("[BATCH] %s: exception at track %d", job_id, i)
        job.update(downloaded=downloaded, failed=failed)

    _invalidate_caches()
    final_status = "cancelled" if (cancel and cancel.is_set()) else "done"
    job.update(
        status=final_status, downloaded=downloaded, failed=failed,
        results=dl_results, elapsed_s=round(time.time() - job["started"], 1),
    )
    _cancel_events.pop(job_id, None)


async def _upgrade_job(job_id: str):
    job = _jobs[job_id]
    try:
        def on_progress(done, total):
            job.update(scanned=done, total=total)

        upgrades = await upgrade_scan(MUSIC_DIR, SLSKD_URL, SLSKD_KEY, on_progress=on_progress)
        job["candidates"] = len(upgrades)
        job["status"] = "upgrading"

        upgraded = 0
        failed = 0
        for u in upgrades:
            try:
                dl = await upgrade_download(u, MUSIC_DIR, SLSKD_URL, SLSKD_KEY)
                if dl:
                    upgraded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            job.update(upgraded=upgraded, failed=failed)
            await asyncio.sleep(3)

        _invalidate_caches()
        job.update(status="done", upgraded=upgraded, failed=failed,
                   elapsed_s=round(time.time() - job["started"], 1))
    except Exception as e:
        job.update(status="failed", error=str(e))


async def _recommend_job(job_id: str, provider: str, api_key: str, model: str, count: int):
    """Background job for AI recommendation generation."""
    job = _jobs[job_id]
    try:
        # Step 1: Get recommendations from LLM
        recs = await get_recommendations(
            MUSIC_DIR, provider, api_key,
            model=model or None, count=count,
        )
        job["recommended"] = len(recs)
        job["status"] = "downloading"
        job["tracks"] = [f"{r['artist']} - {r['title']}" for r in recs]

        if not recs:
            job.update(status="done", message="No recommendations generated")
            return

        # Step 2: Build genre map from AI response
        genre_map = {}
        for rec in recs:
            key = f"{rec.get('artist','')} - {rec.get('title','')}"
            if rec.get("genre"):
                genre_map[key] = rec["genre"]

        # Step 3: Search all tracks
        results = await search_batch(recs, slskd_url=SLSKD_URL, slskd_key=SLSKD_KEY)

        # Step 4: Download to _recommendations dir
        rec_dir = os.path.join(MUSIC_DIR, "_recommendations")
        os.makedirs(rec_dir, exist_ok=True)

        downloaded_paths = []
        for r in results:
            if not r.best:
                continue
            try:
                dl = await download_track(r, rec_dir, SLSKD_URL, SLSKD_KEY)
                if dl and dl.path:
                    downloaded_paths.append(dl.path)
                    job["downloaded"] = len(downloaded_paths)
                    # Tag genre from AI
                    key = f"{r.artist} - {r.title}"
                    genre = genre_map.get(key, "")
                    if genre:
                        from engine import tag_flac as _tag
                        _tag(dl.path, extra_meta={"genre": genre})
            except Exception:
                pass

        # Step 4: Create Navidrome playlist (wait for scan to index new files)
        playlist_id = None
        if downloaded_paths and NAVIDROME_USER and NAVIDROME_PASSWORD:
            # Navidrome scans every 1m, wait for it to pick up new files
            logger.info("[RECOMMEND] Waiting 75s for Navidrome to index %d new tracks...", len(downloaded_paths))
            await asyncio.sleep(75)
            playlist_id = await create_navidrome_playlist(
                NAVIDROME_URL, NAVIDROME_USER, NAVIDROME_PASSWORD,
                downloaded_paths, MUSIC_DIR,
            )
            job["playlist_id"] = playlist_id

        _invalidate_caches()
        _recommend_status["last_run"] = time.time()
        _recommend_status["last_result"] = {
            "recommended": len(recs), "downloaded": len(downloaded_paths),
            "playlist_id": playlist_id, "timestamp": time.time(),
        }

        not_found = sum(1 for r in results if not r.best)
        job.update(
            status="done",
            downloaded=len(downloaded_paths),
            not_found=not_found,
            playlist_id=playlist_id,
            elapsed_s=round(time.time() - job["started"], 1),
        )
        logger.info("[RECOMMEND] Done: %d/%d downloaded, playlist=%s",
                     len(downloaded_paths), len(recs), playlist_id)
    except Exception as e:
        job.update(status="failed", error=str(e))
        logger.exception("[RECOMMEND] Job %s failed", job_id)


def _count_flacs() -> int:
    try:
        return len(list(Path(MUSIC_DIR).rglob("*.flac")))
    except Exception:
        return 0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8844)
