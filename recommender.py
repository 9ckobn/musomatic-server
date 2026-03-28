"""
AI-powered music recommendation engine.

Analyzes user's library, asks an LLM for recommendations,
downloads them into a special playlist, auto-cleans after 24h.
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from mutagen.flac import FLAC

logger = logging.getLogger("recommender")

PROXY_URL = os.getenv("PROXY_URL", "")

# Supported LLM providers
PROVIDERS = {
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": "gemini-2.0-flash",
        "auth": "query",  # API key in URL param
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "auth": "Bearer",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "auth": "Bearer",
    },
    "claude": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-20250514",
        "auth": "x-api-key",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "auth": "Bearer",
    },
}

RECOMMEND_COUNT = int(os.getenv("RECOMMEND_COUNT", "40"))
RECOMMEND_PLAYLIST = "AI Recommendations"
RECOMMEND_CLEANUP_HOURS = int(os.getenv("RECOMMEND_CLEANUP_HOURS", "24"))
RECOMMEND_HISTORY_FILE = os.getenv("RECOMMEND_HISTORY_FILE", "/music/.recommend_history.json")

# Cooldown range: don't re-recommend a track for 7-21 days (random per track)
HISTORY_COOLDOWN_MIN = 7 * 86400   # 7 days
HISTORY_COOLDOWN_MAX = 21 * 86400  # 21 days


def _load_history() -> dict:
    """Load recommendation history. Format: {"artist - title": timestamp}"""
    try:
        with open(RECOMMEND_HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_history(history: dict):
    """Save recommendation history."""
    try:
        with open(RECOMMEND_HISTORY_FILE, "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.warning("[RECOMMEND] Failed to save history")


def _scan_library_for_ai(music_dir: str) -> list[dict]:
    """Get compact track list for the LLM prompt."""
    tracks = []
    for f in sorted(Path(music_dir).rglob("*.flac")):
        try:
            audio = FLAC(str(f))
            artist = (audio.get("artist") or [""])[0]
            title = (audio.get("title") or [""])[0]
            if artist and title:
                tracks.append({"artist": artist, "title": title})
        except Exception:
            continue
    return tracks


def _build_prompt(library: list[dict], count: int = 30, history: dict | None = None) -> str:
    """Build the recommendation prompt with diversity emphasis."""
    import random

    by_artist: dict[str, list[str]] = {}
    for t in library:
        by_artist.setdefault(t["artist"], []).append(t["title"])

    lib_text = "\n".join(
        f"- {artist}: {', '.join(titles)}"
        for artist, titles in sorted(by_artist.items())
    )

    # Build "already recommended" exclusion list from history
    exclude_text = ""
    if history:
        now = time.time()
        active = [k for k, ts in history.items()
                  if now - ts < HISTORY_COOLDOWN_MAX]
        if active:
            sample = random.sample(active, min(len(active), 50))
            exclude_text = "\n\nDo NOT recommend these (recently suggested):\n" + \
                           "\n".join(f"- {t}" for t in sample)

    return f"""You are a cutting-edge music discovery AI for an audiophile who's tired of hearing the same songs.

User's library ({len(by_artist)} artists, {sum(len(v) for v in by_artist.values())} tracks):

{lib_text}

Generate exactly {count} track recommendations following this MIX:

🔥 ~5% ICONIC DEEP CUTS — cult classics that real fans know but aren't overplayed mainstream hits
🆕 ~40% FRESH DISCOVERIES — released in the last 3 years, across all genres
📈 ~15% TRENDING NOW — songs currently viral on TikTok, Reels, trending on Spotify/Apple Music in 2024-2025
🌍 ~20% INTERNATIONAL — non-English: Japanese (J-Rock, J-Pop, City Pop), Korean, French, German, Russian, Scandinavian, Latin American
🎵 ~20% PERFECT MATCHES — great tracks from any era/genre that match the user's vibe but they probably haven't heard

HARD RULES:
- Maximum 1 track per artist
- NO tracks already in the user's library
- NO obvious mega-hits (nothing with 1B+ streams that everyone knows)
- NO remixes, live versions, covers, karaoke
- Every track must be a real, released studio recording findable on Tidal/Spotify
- Include the genre for each track
{exclude_text}

Respond ONLY with a JSON array:
[{{"artist": "Artist Name", "title": "Track Title", "genre": "genre tag"}}, ...]"""


def _parse_truncated_json(raw: str) -> list[dict]:
    """Salvage complete JSON objects from a truncated array response."""
    # Find all complete {...} objects using a simple brace-matching approach
    results = []
    i = raw.find('[')
    if i < 0:
        return results
    i += 1
    while i < len(raw):
        start = raw.find('{', i)
        if start < 0:
            break
        depth = 0
        end = start
        for j in range(start, len(raw)):
            if raw[j] == '{':
                depth += 1
            elif raw[j] == '}':
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if depth != 0:
            break  # incomplete object
        try:
            obj = json.loads(raw[start:end])
            if isinstance(obj, dict) and "artist" in obj and "title" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
        i = end
    logger.info("[LLM] Salvaged %d tracks from truncated JSON response", len(results))
    return results


def _llm_client(timeout: int = 90) -> httpx.AsyncClient:
    """Create httpx client for LLM calls, with optional proxy."""
    opts = dict(timeout=timeout, follow_redirects=True)
    if PROXY_URL:
        opts["proxy"] = PROXY_URL
    return httpx.AsyncClient(**opts)


async def _call_openai_compatible(
    url: str, model: str, api_key: str, prompt: str, timeout: int = 90,
) -> str:
    """Call OpenAI-compatible API (OpenAI, DeepSeek, OpenRouter)."""
    for attempt in range(3):
        async with _llm_client(timeout) as c:
            r = await c.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 16384,
                    "reasoning": {"effort": "none"},
                },
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 20
                logger.warning("[LLM] Rate limited (OpenAI-compat), retry in %ds (attempt %d/3)", wait, attempt + 1)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content
            # Log unexpected response
            logger.error("[LLM] Empty content: %s", json.dumps(data)[:500])
            if data.get("error"):
                raise ValueError(f"LLM error: {data['error']}")
            await asyncio.sleep(10)
    raise ValueError("LLM returned empty content after 3 attempts")


async def _call_claude(
    url: str, model: str, api_key: str, prompt: str, timeout: int = 90,
) -> str:
    """Call Anthropic Claude API."""
    async with _llm_client(timeout) as c:
        r = await c.post(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["content"][0]["text"]


async def _call_gemini(
    url: str, model: str, api_key: str, prompt: str, timeout: int = 90,
) -> str:
    """Call Google Gemini API (free tier) with retry on 429."""
    endpoint = url.replace("{model}", model)
    for attempt in range(5):
        async with _llm_client(timeout) as c:
            r = await c.post(
                endpoint,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.8,
                        "maxOutputTokens": 4096,
                    },
                },
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 30  # 30, 60, 90, 120, 150s
                logger.warning("[LLM] Gemini rate limited, retry in %ds (attempt %d/5)", wait, attempt + 1)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    raise httpx.HTTPStatusError("Gemini rate limit exceeded after 5 retries", request=r.request, response=r)


async def get_recommendations(
    music_dir: str,
    provider: str,
    api_key: str,
    model: str | None = None,
    count: int | None = None,
) -> list[dict]:
    """Analyze library and get AI recommendations.

    Returns list of {"artist": ..., "title": ...} dicts.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    use_model = model or config["model"]
    use_count = count or RECOMMEND_COUNT

    # Scan library
    library = _scan_library_for_ai(music_dir)
    if not library:
        raise ValueError("Library is empty — nothing to analyze")

    # Load recommendation history
    history = _load_history()

    logger.info("[RECOMMEND] Generating: %d tracks in library, %d in history, provider=%s, model=%s",
                len(library), len(history), provider, use_model)

    prompt = _build_prompt(library, use_count, history)

    # Call LLM
    if provider == "claude":
        raw = await _call_claude(config["url"], use_model, api_key, prompt)
    elif provider == "gemini":
        raw = await _call_gemini(config["url"], use_model, api_key, prompt)
    else:
        raw = await _call_openai_compatible(config["url"], use_model, api_key, prompt)

    # Parse JSON from response (handle markdown code blocks)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

    try:
        tracks = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                tracks = json.loads(match.group())
            except json.JSONDecodeError:
                # Truncated JSON — salvage complete entries
                tracks = _parse_truncated_json(match.group())
        else:
            # Maybe truncated — try salvaging from first [
            idx = raw.find('[')
            if idx >= 0:
                tracks = _parse_truncated_json(raw[idx:])
            else:
                raise ValueError(f"LLM returned invalid JSON: {raw[:200]}")

    if not tracks:
        raise ValueError(f"LLM returned no parseable tracks: {raw[:200]}")

    # Validate structure and filter out recently recommended
    now = time.time()
    valid = []
    for t in tracks:
        if isinstance(t, dict) and "artist" in t and "title" in t:
            key = f"{t['artist']} - {t['title']}"
            prev_ts = history.get(key, 0)
            import random
            cooldown = random.randint(HISTORY_COOLDOWN_MIN, HISTORY_COOLDOWN_MAX)
            if now - prev_ts > cooldown:
                entry = {"artist": str(t["artist"]), "title": str(t["title"])}
                if t.get("genre"):
                    entry["genre"] = str(t["genre"])
                valid.append(entry)

    # Record all recommendations in history
    for t in valid:
        history[f"{t['artist']} - {t['title']}"] = now
    _save_history(history)

    logger.info("[RECOMMEND] Got %d valid recs (%d after history filter)", len(tracks), len(valid))
    return valid[:use_count]


# ─── Navidrome Playlist Integration ─────────────────────────────

async def _navidrome_api(
    base_url: str, user: str, password: str,
    endpoint: str, params: dict | None = None,
) -> dict:
    """Call Navidrome's Subsonic API."""
    import hashlib
    salt = os.urandom(6).hex()
    token = hashlib.md5(f"{password}{salt}".encode()).hexdigest()

    base_params = {
        "u": user, "t": token, "s": salt,
        "v": "1.16.1", "c": "musomatic", "f": "json",
    }
    if params:
        base_params.update(params)

    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{base_url}/rest/{endpoint}", params=base_params)
        r.raise_for_status()
        data = r.json()
        return data.get("subsonic-response", data)


async def create_navidrome_playlist(
    navidrome_url: str,
    navidrome_user: str,
    navidrome_password: str,
    track_paths: list[str],
    music_dir: str,
    playlist_name: str = RECOMMEND_PLAYLIST,
) -> str | None:
    """Create a playlist in Navidrome with the given tracks.

    Returns playlist ID or None.
    """
    try:
        # First, get all songs from Navidrome to match by path
        # Use search to find each track
        song_ids = []

        for path in track_paths:
            # Extract artist and title from path for search
            fname = Path(path).stem  # "Artist - Title"
            parts = fname.split(" - ", 1)
            if len(parts) == 2:
                query = f"{parts[0]} {parts[1]}"
            else:
                query = fname

            result = await _navidrome_api(
                navidrome_url, navidrome_user, navidrome_password,
                "search3", {"query": query, "songCount": 5, "albumCount": 0, "artistCount": 0},
            )

            songs = result.get("searchResult3", {}).get("song", [])
            if songs:
                song_ids.append(songs[0]["id"])

        if not song_ids:
            logger.warning("[RECOMMEND] No tracks matched in Navidrome")
            return None

        # Delete existing recommendation playlist if it exists
        playlists_resp = await _navidrome_api(
            navidrome_url, navidrome_user, navidrome_password,
            "getPlaylists", {},
        )
        for pl in playlists_resp.get("playlists", {}).get("playlist", []):
            if pl.get("name") == playlist_name:
                await _navidrome_api(
                    navidrome_url, navidrome_user, navidrome_password,
                    "deletePlaylist", {"id": pl["id"]},
                )
                logger.info("[RECOMMEND] Deleted old playlist: %s", pl["id"])

        # Create new playlist with songs
        params = {"name": playlist_name}
        for i, sid in enumerate(song_ids):
            params[f"songId[{i}]"] = sid  # Subsonic API uses indexed params

        result = await _navidrome_api(
            navidrome_url, navidrome_user, navidrome_password,
            "createPlaylist", params,
        )

        playlist = result.get("playlist", {})
        pid = playlist.get("id", "")
        logger.info("[RECOMMEND] Created playlist '%s' with %d tracks (id=%s)", playlist_name, len(song_ids), pid)
        return pid

    except Exception:
        logger.exception("[RECOMMEND] Failed to create playlist")
        return None


async def cleanup_recommendations(
    music_dir: str,
    navidrome_url: str,
    navidrome_user: str,
    navidrome_password: str,
    playlist_name: str = RECOMMEND_PLAYLIST,
) -> dict:
    """Delete recommendation playlist and its tracks from disk.

    Tracks that were rated/starred by user are moved to main library instead.
    """
    kept = 0
    deleted = 0

    try:
        # Find the recommendation playlist
        playlists_resp = await _navidrome_api(
            navidrome_url, navidrome_user, navidrome_password,
            "getPlaylists", {},
        )
        playlist_id = None
        for pl in playlists_resp.get("playlists", {}).get("playlist", []):
            if pl.get("name") == playlist_name:
                playlist_id = pl["id"]
                break

        if not playlist_id:
            return {"status": "no_playlist", "kept": 0, "deleted": 0}

        # Get playlist tracks with their ratings
        pl_resp = await _navidrome_api(
            navidrome_url, navidrome_user, navidrome_password,
            "getPlaylist", {"id": playlist_id},
        )
        entries = pl_resp.get("playlist", {}).get("entry", [])

        rec_dir = Path(music_dir) / "_recommendations"
        for entry in entries:
            starred = entry.get("starred")
            rating = entry.get("userRating", 0)
            path_str = entry.get("path", "")
            title = entry.get("title", "")
            artist = entry.get("artist", "")

            # Find the actual file
            full_path = None
            if path_str:
                candidate = Path(music_dir) / path_str
                if candidate.exists():
                    full_path = candidate
            if not full_path:
                for f in rec_dir.rglob("*.flac"):
                    try:
                        audio = FLAC(str(f))
                        if (audio.get("title") or [""])[0] == title:
                            full_path = f
                            break
                    except Exception:
                        continue

            if not full_path or not full_path.exists():
                continue

            if starred or rating >= 3:
                # User liked it — move to main library
                safe_a = re.sub(r'[<>:"/\\|?*]', '_', artist).strip() or "Unknown"
                safe_t = re.sub(r'[<>:"/\\|?*]', '_', title).strip() or "Unknown"
                dest_dir = Path(music_dir) / safe_a
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{safe_a} - {safe_t}.flac"
                try:
                    full_path.rename(dest)
                    kept += 1
                    logger.info("[CLEANUP] Kept (rated): %s - %s → %s", artist, title, dest)
                except Exception:
                    logger.warning("[CLEANUP] Failed to move: %s", full_path)
            else:
                # Not rated — delete
                try:
                    full_path.unlink()
                    deleted += 1
                except Exception:
                    pass

        # Clean up empty _recommendations dir
        if rec_dir.exists():
            for d in sorted(rec_dir.rglob("*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            if rec_dir.exists() and not any(rec_dir.iterdir()):
                rec_dir.rmdir()

        # Delete the playlist from Navidrome
        await _navidrome_api(
            navidrome_url, navidrome_user, navidrome_password,
            "deletePlaylist", {"id": playlist_id},
        )
        logger.info("[CLEANUP] Done: kept %d (rated), deleted %d", kept, deleted)

    except Exception:
        logger.exception("[CLEANUP] Failed")

    return {"status": "done", "kept": kept, "deleted": deleted}
