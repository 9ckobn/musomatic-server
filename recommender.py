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

# Supported LLM providers
PROVIDERS = {
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
        "model": "deepseek/deepseek-chat",
        "auth": "Bearer",
    },
}

RECOMMEND_COUNT = int(os.getenv("RECOMMEND_COUNT", "30"))
RECOMMEND_PLAYLIST = "AI Recommendations"
RECOMMEND_CLEANUP_HOURS = int(os.getenv("RECOMMEND_CLEANUP_HOURS", "24"))


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


def _build_prompt(library: list[dict], count: int = 30) -> str:
    """Build the recommendation prompt."""
    # Deduplicate and group by artist for a compact representation
    by_artist: dict[str, list[str]] = {}
    for t in library:
        by_artist.setdefault(t["artist"], []).append(t["title"])

    lib_text = "\n".join(
        f"- {artist}: {', '.join(titles)}"
        for artist, titles in sorted(by_artist.items())
    )

    return f"""You are a music recommendation engine for an audiophile.

Here is the user's current music library:

{lib_text}

Based on this collection, recommend exactly {count} tracks that the user would likely enjoy.

Rules:
- Recommend tracks NOT already in the library
- Match the user's taste: similar genres, energy, era
- Mix well-known classics with hidden gems
- Include a variety of artists (don't repeat artists too much)
- Only recommend tracks that exist and are searchable
- NO remixes, live versions, covers, or karaoke

Respond ONLY with a JSON array, no other text:
[{{"artist": "Artist Name", "title": "Track Title"}}, ...]"""


async def _call_openai_compatible(
    url: str, model: str, api_key: str, prompt: str, timeout: int = 60,
) -> str:
    """Call OpenAI-compatible API (OpenAI, DeepSeek, OpenRouter)."""
    async with httpx.AsyncClient(timeout=timeout) as c:
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
                "max_tokens": 4096,
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def _call_claude(
    url: str, model: str, api_key: str, prompt: str, timeout: int = 60,
) -> str:
    """Call Anthropic Claude API."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["content"][0]["text"]


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

    logger.info("Generating recommendations: %d tracks in library, provider=%s, model=%s",
                len(library), provider, use_model)

    prompt = _build_prompt(library, use_count)

    # Call LLM
    if provider == "claude":
        raw = await _call_claude(config["url"], use_model, api_key, prompt)
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
            tracks = json.loads(match.group())
        else:
            raise ValueError(f"LLM returned invalid JSON: {raw[:200]}")

    # Validate structure
    valid = []
    for t in tracks:
        if isinstance(t, dict) and "artist" in t and "title" in t:
            valid.append({"artist": str(t["artist"]), "title": str(t["title"])})

    logger.info("Got %d valid recommendations from LLM", len(valid))
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
            logger.warning("No tracks matched in Navidrome")
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
                logger.info("Deleted old recommendation playlist: %s", pl["id"])

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
        logger.info("Created playlist '%s' with %d tracks (id=%s)", playlist_name, len(song_ids), pid)
        return pid

    except Exception:
        logger.exception("Failed to create Navidrome playlist")
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
                    logger.info("Kept (rated): %s - %s → %s", artist, title, dest)
                except Exception:
                    logger.warning("Failed to move rated track: %s", full_path)
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
        logger.info("Cleanup done: kept %d (rated), deleted %d", kept, deleted)

    except Exception:
        logger.exception("Recommendation cleanup failed")

    return {"status": "done", "kept": kept, "deleted": deleted}
