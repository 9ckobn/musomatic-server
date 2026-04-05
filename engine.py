"""
Multi-source lossless music engine.

Searches Tidal (via Monochrome API) + Soulseek (via slskd) in parallel,
picks best quality, downloads FLAC to organized library.
"""
import asyncio
import json
import base64
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from mutagen.flac import FLAC, Picture

logger = logging.getLogger(__name__)

MONOCHROME_API = os.getenv("MONOCHROME_API", "https://api.monochrome.tf")
PROXY_URL = os.getenv("PROXY_URL", "")


def _proxy_client(timeout=15, **kwargs) -> httpx.AsyncClient:
    """Create httpx client that routes through the configured proxy."""
    transport = httpx.AsyncHTTPTransport(retries=3)
    opts = dict(timeout=timeout, follow_redirects=True, transport=transport, **kwargs)
    if PROXY_URL:
        opts["proxy"] = PROXY_URL
        opts.pop("transport", None)  # proxy incompatible with custom transport
    return httpx.AsyncClient(**opts)

_STOP_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "by",
    "for", "and", "or", "is", "it",
}


def _clean_query(artist: str, title: str) -> str:
    """Build a clean search query, stripping noise."""
    title = re.sub(r'\s*[\(\[].*?[\)\]]', '', title)
    title = re.sub(r'\s*(feat\.?|ft\.?|featuring)\s+.*', '', title, flags=re.IGNORECASE)
    if '—' in title or '–' in title:
        title = re.sub(r'^.*?\s*[—–-]\s+', '', title)
    artist = re.split(r'\s*[×&,]\s*', artist)[0].strip()
    artist = re.sub(r'\s*[\(].*', '', artist)
    title_words = title.split()
    if len(title_words) >= 4:
        title_clean = ' '.join(w for w in title_words if w.lower() not in _STOP_WORDS)
        if title_clean:
            title = title_clean
    return f"{artist} {title}".strip()


_PENALTY_WORDS = {
    "live", "remix", "cover", "acoustic", "demo", "instrumental",
    "karaoke", "tribute", "unplugged",
    "radio", "edit", "version", "mix", "session", "rehearsal",
    "bootleg", "concert", "performed", "originally",
}


# ─── Non-Original Detection ─────────────────────────────────────

_NON_ORIGINAL_RE = re.compile(
    r'(?i)'
    r'(?:'
    # Live versions
    r'\b(?:live\s+(?:at|from|in|version))\b'
    r'|[\(\[]\s*live(?:\s+(?:at|from|in)\b[^\)\]]*)?[\)\]]'
    r'|\blive\s+(?:recording|session|performance|concert|album|spirits)\b'
    r'|\b(?:unplugged|mtv\s+unplugged)\b'
    # Remixes
    r'|[\(\[]\s*[^\)\]]*\s*remix\s*[\)\]]'
    r'|[\(\[]\s*[^\)\]]*\s*mix\s*[\)\]]'
    r'|\bremixed\s+by\b'
    # Edits
    r'|[\(\[]\s*(?:radio|club|extended|rave\s+music)\s+edit\s*[\)\]]'
    # Acoustic
    r'|[\(\[]\s*acoustic\s*[\)\]]'
    r'|\bacoustic\s+version\b'
    # Demos
    r'|[\(\[]\s*demo\s*[\)\]]'
    r'|\bdemo\s+version\b'
    r'|\boriginal\s+demo\b'
    # Karaoke / instrumental / backing
    r'|\bkaraoke\b'
    r'|\binstrumental\s+version\b'
    r'|\bbacking\s+track\b'
    # Covers / tributes
    r'|\boriginally\s+performed\b'
    r'|\btribute\b'
    r'|\bin\s+the\s+style\s+of\b'
    r'|\bmade\s+famous\s+by\b'
    r'|\bmade\s+popular\s+by\b'
    r'|\bcover\b'
    # Movie / soundtrack markers
    r'|[\(\[]\s*from\s+["_\u201c][^\)\]]*[\)\]]'
    r'|\bsoundtrack\b'
    # Version variants
    r'|[\(\[]\s*(?:karaoke|acoustic|demo|radio|club|instrumental|vocal)\s+version\s*[\)\]]'
    # Remastered — NOT filtered: remasters are the same original track
    # r'|[\(\[]\s*(?:\d{4}\s*[-\u2013\u2014]\s*)?remaster(?:ed)?\s*[\)\]]'
    # Russian markers
    r'|[\(\[]\s*\u0412\u0435\u0440\u0441\u0438\u044f\s+[^\)\]]*[\)\]]'
    r')'
)

_REMASTER_RE = re.compile(
    r'[\(\[]\s*(?:\d{4}\s*[-\u2013\u2014]\s*)?remaster(?:ed)?\s*[\)\]]'
    r'|[\(\[]\s*\d{4}\s+remaster(?:ed)?\s*[\)\]]',
    re.IGNORECASE,
)


def _is_non_original(query_title: str, result_title: str,
                     result_version: str = "", result_album: str = "") -> bool:
    """Check if a result is a non-original version (remix, live, etc.).

    Returns False if the QUERY itself asks for a non-original (user wants it).
    Returns True if the result title/version/album contains non-original markers.
    """
    if _NON_ORIGINAL_RE.search(query_title):
        return False
    for text in (result_title, result_version, result_album):
        if text and _NON_ORIGINAL_RE.search(text):
            return True
    return False


def _fuzzy_match(a: str, b: str) -> float:
    a = re.sub(r'[^\w\s]', '', a.lower()).strip()
    b = re.sub(r'[^\w\s]', '', b.lower()).strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    words_a = set(a.split()) - _STOP_WORDS
    words_b = set(b.split()) - _STOP_WORDS
    if not words_a or not words_b:
        words_a = set(a.split())
        words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _has_penalty_words(title: str) -> bool:
    words = set(re.sub(r'[^\w\s]', '', title.lower()).split())
    return bool(words & _PENALTY_WORDS)


def _title_match_strict(query_title: str, result_title: str) -> bool:
    q_has = _has_penalty_words(query_title)
    r_has = _has_penalty_words(result_title)
    if r_has and not q_has:
        return False
    return True


class Quality(IntEnum):
    NONE = 0
    MP3_128 = 1
    MP3_320 = 2
    FLAC_16 = 3
    FLAC_24 = 4
    FLAC_24_HI = 5


@dataclass
class SourceResult:
    source: str
    artist: str
    title: str
    album: str = ""
    quality: Quality = Quality.NONE
    quality_label: str = ""
    sample_rate: int = 0
    bit_depth: int = 0
    bitrate: int = 0
    size_mb: float = 0
    track_id: str = ""
    peer: str = ""
    filename: str = ""
    download_url: str = ""
    available: bool = False
    tags: list = field(default_factory=list)
    popularity: int = 0
    version: str = ""

    @property
    def is_hires(self) -> bool:
        return self.bit_depth >= 24

    @property
    def is_lossless(self) -> bool:
        return self.quality >= Quality.FLAC_16

    def to_dict(self) -> dict:
        return {
            "source": self.source, "artist": self.artist, "title": self.title,
            "album": self.album, "quality": self.quality.name,
            "quality_label": self.quality_label, "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth, "size_mb": round(self.size_mb, 1),
            "is_hires": self.is_hires, "peer": self.peer,
            "popularity": self.popularity, "version": self.version,
        }


@dataclass
class UnifiedResult:
    query: str
    artist: str
    title: str
    sources: list[SourceResult] = field(default_factory=list)

    @property
    def best(self) -> SourceResult | None:
        avail = [s for s in self.sources if s.available]
        if not avail:
            return None

        def score(s: SourceResult) -> tuple:
            if s.source == "soulseek":
                fname = s.title.lower()
                a_in = _fuzzy_match(fname, self.artist) if self.artist else 0.3
                t_in = _fuzzy_match(fname, self.title) if self.title else 0.3
                both_match = max(a_in, t_in) >= 0.3
                no_penalty = True
                relevance = 0.5
                is_original = not _is_non_original(self.title, fname)
            else:
                if self.artist:
                    a_match = _fuzzy_match(s.artist, self.artist)
                else:
                    a_match = _fuzzy_match(s.artist, self.query) if self.query else 0.3
                t_match = _fuzzy_match(s.title, self.title) if self.title else 0.5
                both_match = (a_match >= 0.3 if self.artist else a_match >= 0.2) and t_match >= 0.3
                no_penalty = _title_match_strict(self.title, s.title)
                relevance = a_match + t_match
                is_original = not _is_non_original(self.title, s.title, s.version, s.album)
            pop = s.popularity / 100.0
            return (
                both_match and no_penalty and is_original,
                is_original,
                no_penalty,
                pop >= 0.5,
                relevance >= 1.5,
                pop,
                relevance,
                s.bit_depth >= 24,
                s.quality,
                s.sample_rate,
            )

        best = max(avail, key=score)
        if not score(best)[0]:
            return None
        return best

    @property
    def all_ranked(self) -> list[SourceResult]:
        avail = [s for s in self.sources if s.available]

        def score(s: SourceResult) -> tuple:
            if s.source == "soulseek":
                fname = s.title.lower()
                a_in = _fuzzy_match(fname, self.artist) if self.artist else 0.3
                t_in = _fuzzy_match(fname, self.title) if self.title else 0.3
                both_match = max(a_in, t_in) >= 0.3
                no_penalty = True
                relevance = 0.5
                is_original = not _is_non_original(self.title, fname)
            else:
                if self.artist:
                    a_match = _fuzzy_match(s.artist, self.artist)
                else:
                    a_match = _fuzzy_match(s.artist, self.query) if self.query else 0.3
                t_match = _fuzzy_match(s.title, self.title) if self.title else 0.5
                both_match = (a_match >= 0.3 if self.artist else a_match >= 0.2) and t_match >= 0.3
                no_penalty = _title_match_strict(self.title, s.title)
                relevance = a_match + t_match
                is_original = not _is_non_original(self.title, s.title, s.version, s.album)
            pop = s.popularity / 100.0
            # Tidal HI_RES tracks are prioritised — Monochrome proxy
            # currently only serves streams for HI_RES-tagged content
            tidal_hires = s.source == "tidal" and s.bit_depth >= 24
            return (
                both_match and no_penalty and is_original,
                is_original,
                no_penalty,
                tidal_hires,
                s.bit_depth >= 24,
                pop >= 0.5,
                relevance >= 1.5,
                pop,
                relevance,
                s.quality,
                s.sample_rate,
            )

        return sorted([s for s in avail if score(s)[0]], key=score, reverse=True)

    def to_dict(self) -> dict:
        best = self.best
        return {
            "artist": self.artist, "title": self.title, "query": self.query,
            "best": best.to_dict() if best else None,
            "source_count": len(self.sources),
            "has_hires": best.is_hires if best else False,
            "has_lossless": best.is_lossless if best else False,
        }


@dataclass
class DownloadResult:
    path: str
    bit_depth: int
    sample_rate: int
    stream_type: str
    artist: str = ""
    title: str = ""
    album: str = ""

    @property
    def quality_tag(self) -> str:
        rate = f"{self.sample_rate / 1000:.1f}kHz"
        if self.bit_depth >= 24:
            return f"Hi-Res {self.bit_depth}bit/{rate}"
        return f"FLAC CD {self.bit_depth}bit/{rate}"

    def to_dict(self) -> dict:
        return {
            "path": self.path, "bit_depth": self.bit_depth,
            "sample_rate": self.sample_rate, "stream_type": self.stream_type,
            "quality_tag": self.quality_tag, "artist": self.artist,
            "title": self.title, "album": self.album,
            "size_mb": round(Path(self.path).stat().st_size / 1_000_000, 1)
            if Path(self.path).exists() else 0,
        }


# ─── MusicBrainz Genre Lookup ────────────────────────────────────

_MB_GENRE_CACHE: dict[str, str] = {}

async def _fetch_genre_musicbrainz(artist: str) -> str:
    """Fetch genre tags from MusicBrainz by artist name. Cached per artist."""
    if artist in _MB_GENRE_CACHE:
        return _MB_GENRE_CACHE[artist]
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://musicbrainz.org/ws/2/artist",
                params={"query": artist, "limit": 1, "fmt": "json"},
                headers={"User-Agent": "musomatic/1.0 (https://github.com/9ckobn/musomatic-server)"},
            )
            if r.status_code == 200:
                artists = r.json().get("artists", [])
                if artists:
                    tags = artists[0].get("tags", [])
                    # Sort by count (most voted), take top genre-like tags
                    genre_tags = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
                    # Filter out non-genre tags (country names, etc.)
                    skip = {"american", "usa", "british", "uk", "américain", "united states",
                            "japanese", "german", "swedish", "korean", "french", "russian",
                            "rock and indie", "seen live", "favorites"}
                    genres = [t["name"] for t in genre_tags
                              if t.get("name", "").lower() not in skip][:3]
                    result = "; ".join(genres) if genres else ""
                    _MB_GENRE_CACHE[artist] = result
                    return result
    except Exception:
        pass
    _MB_GENRE_CACHE[artist] = ""
    return ""


# ─── FLAC Metadata Tagging ──────────────────────────────────────

async def _fetch_track_meta(track_id: str) -> dict:
    """Fetch full track metadata + cover from Tidal.

    Returns dict with: cover_data, date, tracknumber, discnumber,
    isrc, copyright, duration, explicit, artists (list).
    """
    meta = {}
    try:
        async with _proxy_client(timeout=15) as c:
            r = await c.get(f"{MONOCHROME_API}/info/", params={"id": track_id})
            if r.status_code != 200:
                return meta
            data = r.json().get("data", {})

            # Extract metadata fields
            meta["date"] = (data.get("streamStartDate") or "")[:10]  # "2018-05-29"
            meta["tracknumber"] = str(data.get("trackNumber", ""))
            meta["discnumber"] = str(data.get("volumeNumber", ""))
            meta["isrc"] = data.get("isrc", "")
            meta["copyright"] = data.get("copyright", "")
            meta["explicit"] = "1" if data.get("explicit") else ""
            meta["bpm"] = str(data.get("bpm", "")) if data.get("bpm") else ""

            # All artists
            artists = data.get("artists", [])
            if artists:
                meta["artists"] = [a.get("name", "") for a in artists if a.get("name")]

            # Album info
            album = data.get("album", {})
            meta["album"] = album.get("title", "")

            # Cover art
            cover_id = album.get("cover", "")
            if cover_id:
                cover_url = f"https://resources.tidal.com/images/{cover_id.replace('-', '/')}/1280x1280.jpg"
                resp = await c.get(cover_url)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    meta["cover_data"] = resp.content
    except Exception:
        logger.warning("Failed to fetch metadata for track %s", track_id, exc_info=True)
    return meta


def tag_flac(
    path: str,
    artist: str = "",
    title: str = "",
    album: str = "",
    cover_data: bytes | None = None,
    extra_meta: dict | None = None,
):
    """Tag FLAC file with metadata.

    extra_meta can contain: date, tracknumber, discnumber, isrc,
    copyright, explicit, bpm, artists (list).
    """
    try:
        audio = FLAC(path)
        if artist:
            audio["ARTIST"] = artist
            audio["ALBUMARTIST"] = artist
        if title:
            audio["TITLE"] = title
        if album:
            audio["ALBUM"] = album

        if extra_meta:
            if extra_meta.get("date"):
                audio["DATE"] = extra_meta["date"]
                year = extra_meta["date"][:4]
                if year.isdigit():
                    audio["YEAR"] = year
            if extra_meta.get("tracknumber"):
                audio["TRACKNUMBER"] = extra_meta["tracknumber"]
            if extra_meta.get("discnumber"):
                audio["DISCNUMBER"] = extra_meta["discnumber"]
            if extra_meta.get("isrc"):
                audio["ISRC"] = extra_meta["isrc"]
            if extra_meta.get("copyright"):
                audio["COPYRIGHT"] = extra_meta["copyright"]
            if extra_meta.get("genre"):
                audio["GENRE"] = extra_meta["genre"]
            # Multiple artists
            if extra_meta.get("artists") and len(extra_meta["artists"]) > 1:
                audio["ARTIST"] = extra_meta["artists"]
                audio["ALBUMARTIST"] = extra_meta["artists"][0]

        if cover_data:
            pic = Picture()
            pic.type = 3
            pic.mime = "image/jpeg"
            pic.desc = "Cover"
            pic.data = cover_data
            audio.clear_pictures()
            audio.add_picture(pic)
        audio.save()
    except Exception:
        logger.warning("Failed to tag FLAC file %s", path, exc_info=True)


def _normalize_for_match(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*&,]', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def _find_existing(music_dir: str, artist: str, title: str) -> str | None:
    """Check if a track with similar artist+title already exists.
    Checks FLAC metadata first, falls back to filename."""
    norm_a = _normalize_for_match(artist)
    norm_t = _normalize_for_match(title)
    for f in Path(music_dir).rglob("*.flac"):
        try:
            audio = FLAC(str(f))
            meta_artist = _normalize_for_match((audio.get("artist") or [""])[0])
            meta_title = _normalize_for_match((audio.get("title") or [""])[0])
            if meta_title and meta_artist:
                if (norm_t in meta_title or meta_title in norm_t) and \
                   (not norm_a or norm_a.split()[0] in meta_artist):
                    return str(f)
                continue
        except Exception:
            pass
        fname = _normalize_for_match(f.stem)
        if norm_t in fname and (not norm_a or norm_a.split()[0] in fname):
            return str(f)
    return None


def _organize_path(music_dir: str, artist: str, title: str) -> str:
    safe_a = re.sub(r'[<>:"/\\|?*]', '_', artist).strip() or "Unknown Artist"
    safe_t = re.sub(r'[<>:"/\\|?*]', '_', title).strip() or "Unknown"
    artist_dir = Path(music_dir) / safe_a
    artist_dir.mkdir(parents=True, exist_ok=True)
    return str(artist_dir / f"{safe_a} - {safe_t}.flac")


# ─── Tidal (Monochrome API) ─────────────────────────────────────

async def tidal_search(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 10,
    verify_quality: bool = False,
) -> list[SourceResult]:
    results = []
    try:
        r = await client.get(f"{MONOCHROME_API}/search/", params={"s": query, "limit": limit})
        if r.status_code != 200:
            return results
        items = r.json().get("data", {}).get("items", [])
        for item in items:
            tags = item.get("mediaMetadata", {}).get("tags", [])
            aq = item.get("audioQuality", "")
            artist = item.get("artist", {}).get("name", "")
            pop = item.get("popularity", 0)
            ver = item.get("version", "") or ""
            if "HIRES_LOSSLESS" in tags:
                quality, label = Quality.FLAC_24_HI, "Hi-Res 24bit"
            elif aq == "LOSSLESS" or "LOSSLESS" in tags:
                quality, label = Quality.FLAC_16, "FLAC CD"
            elif aq == "HIGH":
                quality, label = Quality.MP3_320, "AAC 320"
            else:
                quality, label = Quality.MP3_128, "LOW"
            results.append(SourceResult(
                source="tidal", artist=artist, title=item.get("title", ""),
                album=item.get("album", {}).get("title", ""),
                quality=quality, quality_label=label,
                track_id=str(item.get("id", "")), available=True, tags=tags,
                popularity=pop, version=ver,
            ))
    except Exception:
        pass

    if verify_quality and results:
        hires_candidates = [r for r in results if "HIRES_LOSSLESS" in r.tags]
        if hires_candidates:
            tasks = [_verify_tidal_quality(client, r) for r in hires_candidates[:5]]
            await asyncio.gather(*tasks, return_exceptions=True)
        verified_hires = [r for r in results if r.bit_depth >= 24]
        if not verified_hires:
            await _verify_tidal_quality(client, results[0])
    return results


async def _verify_tidal_quality(client: httpx.AsyncClient, result: SourceResult):
    try:
        r = await client.get(
            f"{MONOCHROME_API}/track/",
            params={"id": result.track_id, "quality": "HI_RES_LOSSLESS"},
        )
        if r.status_code != 200:
            return
        data = r.json().get("data", {})
        bd = data.get("bitDepth", 16)
        sr = data.get("sampleRate", 44100)
        manifest_b64 = data.get("manifest", "")
        raw = base64.b64decode(manifest_b64) if manifest_b64 else b""
        is_dash = raw.lstrip().startswith(b"<") if raw else False
        if is_dash and bd >= 24:
            result.quality = Quality.FLAC_24_HI
            result.quality_label = f"Hi-Res {bd}bit/{sr / 1000:.1f}kHz"
            result.sample_rate = sr
            result.bit_depth = bd
        else:
            result.quality = Quality.FLAC_16
            result.quality_label = f"FLAC CD 16bit/{sr / 1000:.1f}kHz"
            result.sample_rate = sr
            result.bit_depth = 16
    except Exception:
        pass


async def tidal_get_stream(
    client: httpx.AsyncClient,
    track_id: str,
    quality: str = "HI_RES_LOSSLESS",
) -> dict | None:
    # Always try HI_RES_LOSSLESS first — Monochrome proxy often only serves this tier
    priorities = []
    for q_opt in ["HI_RES_LOSSLESS", quality, "LOSSLESS", "HIGH"]:
        if q_opt not in priorities:
            priorities.append(q_opt)
    for q in priorities:
        try:
            r = await client.get(
                f"{MONOCHROME_API}/track/",
                params={"id": track_id, "quality": q},
            )
            if r.status_code != 200:
                continue
            data = r.json().get("data", {})
            manifest_b64 = data.get("manifest", "")
            if not manifest_b64:
                continue
            raw = base64.b64decode(manifest_b64)
            bit_depth = data.get("bitDepth", 16)
            sample_rate = data.get("sampleRate", 44100)

            if not raw.lstrip().startswith(b"<"):
                manifest = json.loads(raw)
                urls = manifest.get("urls", [])
                if urls:
                    return {
                        "type": "direct", "url": urls[0],
                        "bit_depth": bit_depth, "sample_rate": sample_rate,
                    }
                continue

            root = ET.fromstring(raw.decode())
            for st in root.iter():
                if "SegmentTemplate" not in st.tag:
                    continue
                init_url = st.get("initialization")
                media_tpl = st.get("media")
                start_num = int(st.get("startNumber", "1"))
                if not init_url or not media_tpl or "$Number$" not in media_tpl:
                    continue
                total_segs = 0
                for s_elem in st.iter():
                    tag = s_elem.tag.split("}")[-1] if "}" in s_elem.tag else s_elem.tag
                    if tag == "S":
                        total_segs += 1 + int(s_elem.get("r", "0"))
                if total_segs > 0:
                    return {
                        "type": "dash", "init_url": init_url,
                        "media_template": media_tpl, "start_number": start_num,
                        "total_segments": total_segs,
                        "bit_depth": bit_depth, "sample_rate": sample_rate,
                    }
        except Exception:
            continue
    return None


async def tidal_download(
    track_id: str, output_dir: str,
    artist: str = "", title: str = "",
    album: str = "",
    quality: str = "HI_RES_LOSSLESS",
) -> DownloadResult | None:
    import tempfile

    async with _proxy_client(timeout=httpx.Timeout(180, connect=30)) as c:
        # Fetch full metadata (cover, date, tracknumber, ISRC, etc.)
        track_meta = await _fetch_track_meta(track_id)

        if not artist:
            artist = track_meta.get("artists", ["Unknown"])[0] if track_meta.get("artists") else "Unknown"
        if not title:
            title = track_meta.get("title", "Unknown")
        if not album:
            album = track_meta.get("album", "")

        cover_data = track_meta.pop("cover_data", None)
        fpath = _organize_path(output_dir, artist, title)

        for q in ([quality] if quality == "LOSSLESS" else [quality, "LOSSLESS"]):
            stream = await tidal_get_stream(c, track_id, q)
            if not stream:
                logger.debug("[TIDAL-DL] No stream for %s q=%s", track_id, q)
                continue
            logger.info("[TIDAL-DL] Got %s stream for %s (%dbit/%dHz)",
                        stream["type"], track_id, stream["bit_depth"], stream["sample_rate"])
            tmp_path = None
            try:
                if stream["type"] == "direct":
                    async with c.stream("GET", stream["url"]) as resp:
                        if resp.status_code != 200:
                            logger.warning("[TIDAL-DL] Direct stream %d for %s", resp.status_code, track_id)
                            continue
                        with open(fpath, "wb") as f:
                            async for chunk in resp.aiter_bytes(65536):
                                f.write(chunk)
                    fsize = Path(fpath).stat().st_size
                    if fsize > 1000:
                        tag_flac(fpath, artist=artist, title=title, album=album,
                                 cover_data=cover_data, extra_meta=track_meta)
                        logger.info("[TIDAL-DL] ✅ Direct %s → %s (%.1fMB)", track_id, fpath, fsize/1e6)
                        return DownloadResult(
                            path=fpath, bit_depth=stream["bit_depth"],
                            sample_rate=stream["sample_rate"],
                            stream_type="direct", artist=artist, title=title,
                            album=album,
                        )
                    logger.warning("[TIDAL-DL] File too small: %d bytes", fsize)

                elif stream["type"] == "dash":
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp_path = tmp.name
                        async with c.stream("GET", stream["init_url"]) as resp:
                            if resp.status_code != 200:
                                raise Exception(f"init fail: {resp.status_code}")
                            async for chunk in resp.aiter_bytes(65536):
                                tmp.write(chunk)
                        for i in range(stream["start_number"],
                                       stream["start_number"] + stream["total_segments"]):
                            url = stream["media_template"].replace("$Number$", str(i))
                            async with c.stream("GET", url) as resp:
                                if resp.status_code != 200:
                                    raise Exception(f"seg {i} fail: {resp.status_code}")
                                async for chunk in resp.aiter_bytes(65536):
                                    tmp.write(chunk)

                    tmp_size = Path(tmp_path).stat().st_size
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp_path, "-c:a", "copy", fpath],
                        capture_output=True, text=True,
                    )
                    Path(tmp_path).unlink(missing_ok=True)
                    if result.returncode == 0 and Path(fpath).stat().st_size > 1000:
                        fsize = Path(fpath).stat().st_size
                        tag_flac(fpath, artist=artist, title=title, album=album,
                                 cover_data=cover_data, extra_meta=track_meta)
                        logger.info("[TIDAL-DL] ✅ DASH %s → %s (%.1fMB)", track_id, fpath, fsize/1e6)
                        return DownloadResult(
                            path=fpath, bit_depth=stream["bit_depth"],
                            sample_rate=stream["sample_rate"],
                            stream_type="dash", artist=artist, title=title,
                            album=album,
                        )
                    logger.warning("[TIDAL-DL] ffmpeg failed (rc=%d) or tiny file. mp4=%dB stderr=%s",
                                   result.returncode, tmp_size, result.stderr[:200])
            except Exception:
                logger.warning("[TIDAL-DL] Download failed for %s (q=%s)", track_id, q, exc_info=True)
                Path(fpath).unlink(missing_ok=True)
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
                continue
    return None


# ─── Soulseek (slskd) ───────────────────────────────────────────

async def soulseek_search(
    client: httpx.AsyncClient,
    query: str,
    wait: int = 8,
) -> list[SourceResult]:
    results = []
    queries_to_try = [query]
    # Fallback: title only (last 1-3 words) — Soulseek AND-matches all words
    words = query.split()
    if len(words) >= 3:
        # Try shorter query (last 2 words likely = title)
        short = " ".join(words[-2:])
        if short != query:
            queries_to_try.append(short)

    for q in queries_to_try:
        sid = None
        try:
            r = await client.post("/api/v0/searches", json={"searchText": q})
            if r.status_code not in (200, 201):
                continue
            sid = r.json().get("id")
            if not sid:
                continue
            await asyncio.sleep(wait)
            rr = await client.get(f"/api/v0/searches/{sid}/responses")
            for resp in rr.json():
                username = resp.get("username", "")
                for f in resp.get("files", []):
                    fname = f.get("filename", "")
                    short_name = fname.rsplit("\\", 1)[-1] if "\\" in fname else fname.rsplit("/", 1)[-1]
                    ext = short_name.rsplit(".", 1)[-1].lower() if "." in short_name else ""
                    sz = f.get("size", 0)
                    if ext not in ("flac", "wav", "alac", "ape", "wv"):
                        continue
                    if sz < 5_000_000:
                        continue
                    path_lower = fname.lower()
                    hires_indicators = [
                        "24bit", "24-bit", "24 bit", "hi-res", "hires",
                        "24-96", "24-192", "24-48", "24-44", "[24", "(24",
                    ]
                    is_hires_path = any(ind in path_lower for ind in hires_indicators)
                    bd = f.get("bitDepth") or 0
                    sr = f.get("sampleRate") or 0
                    br = f.get("bitRate") or 0
                    if bd >= 24 or is_hires_path:
                        quality = Quality.FLAC_24_HI if (sr or 44100) >= 88200 else Quality.FLAC_24
                        bit_depth = bd if bd >= 24 else 24
                        label = f"FLAC {bit_depth}bit{'/' + str(sr) + 'Hz' if sr else ''}"
                    else:
                        quality = Quality.FLAC_16
                        bit_depth = bd if bd else 16
                        label = f"FLAC {bit_depth}bit"
                    results.append(SourceResult(
                        source="soulseek", artist="", title=short_name,
                        quality=quality, quality_label=label,
                        sample_rate=sr or 44100, bit_depth=bit_depth,
                        bitrate=br, size_mb=sz / 1_000_000,
                        peer=username, filename=fname, available=True,
                    ))
            await client.delete(f"/api/v0/searches/{sid}")
            sid = None
            if results:
                break  # got results, no need for fallback query
        except Exception:
            pass
        finally:
            if sid:
                try:
                    await client.delete(f"/api/v0/searches/{sid}")
                except Exception:
                    pass

    results.sort(key=lambda r: (r.quality, r.sample_rate, r.size_mb), reverse=True)
    return results[:20]


async def soulseek_download(
    slsk_client: httpx.AsyncClient,
    username: str,
    filename: str,
    music_dir: str,
    bit_depth: int = 16,
    sample_rate: int = 44100,
    artist: str = "",
    title: str = "",
    timeout: int = 60,
    file_size: int = 0,
) -> DownloadResult | None:
    import time as _time
    try:
        r = await slsk_client.post(
            f"/api/v0/transfers/downloads/{username}",
            json=[{"filename": filename, "size": file_size}],
        )
        if r.status_code not in (200, 201):
            return None

        short_fn = filename.rsplit("\\", 1)[-1] if "\\" in filename else filename.rsplit("/", 1)[-1]
        logger.info("[SLSK-DL] Waiting for %s from %s (%s)", short_fn, username, filename)
        queued_polls = 0
        dl_start_time = None  # when InProgress first detected
        for poll in range(timeout // 3):
            await asyncio.sleep(3)
            r = await slsk_client.get(f"/api/v0/transfers/downloads/{username}")
            if r.status_code != 200:
                continue
            data = r.json()
            directories = []
            if isinstance(data, dict):
                directories = data.get("directories", [])
            elif isinstance(data, list):
                directories = data
            found_file = False
            for d in directories:
                files = d.get("files", []) if isinstance(d, dict) else []
                for dl_file in files:
                    if not isinstance(dl_file, dict):
                        continue
                    if dl_file.get("filename") != filename:
                        continue
                    found_file = True
                    state = dl_file.get("state", "")
                    pct = dl_file.get("percentComplete", 0)
                    if poll % 10 == 0:
                        logger.info("[SLSK-DL] %s@%s: state=%s pct=%s%%", short_fn, username, state, pct)
                    if "Completed" in state and "Succeeded" in state:
                        logger.info("[SLSK-DL] ✅ Completed from %s", username)
                        found = _find_downloaded_file(music_dir, short_fn)
                        if found:
                            dest = _organize_path(music_dir, artist, title)
                            # Convert non-FLAC lossless to FLAC
                            ext = found.suffix.lower()
                            if ext in (".wav", ".ape", ".wv", ".alac"):
                                flac_dest = dest  # already ends in .flac
                                try:
                                    proc = await asyncio.create_subprocess_exec(
                                        "ffmpeg", "-y", "-i", str(found),
                                        "-c:a", "flac", "-compression_level", "8",
                                        flac_dest,
                                        stdout=asyncio.subprocess.DEVNULL,
                                        stderr=asyncio.subprocess.DEVNULL,
                                    )
                                    await asyncio.wait_for(proc.wait(), timeout=60)
                                    if proc.returncode == 0 and Path(flac_dest).exists():
                                        found.unlink(missing_ok=True)
                                        logger.info("[SLSK-DL] Converted %s → FLAC", ext)
                                    else:
                                        logger.warning("[SLSK-DL] ffmpeg conversion failed for %s", ext)
                                        found.unlink(missing_ok=True)
                                        return None
                                except Exception:
                                    found.unlink(missing_ok=True)
                                    return None
                            else:
                                try:
                                    shutil.move(str(found), dest)
                                    if found.parent != Path(music_dir) and not any(found.parent.iterdir()):
                                        found.parent.rmdir()
                                except Exception:
                                    dest = str(found)
                            tag_flac(dest, artist=artist, title=title, album="")
                            return DownloadResult(
                                path=dest, bit_depth=bit_depth,
                                sample_rate=sample_rate, stream_type="soulseek",
                                artist=artist, title=title,
                            )
                        logger.warning("[SLSK-DL] Completed but file not found on disk: %s", short_fn)
                        return None
                    elif "Errored" in state or "Cancelled" in state or "Aborted" in state or "TimedOut" in state or "Rejected" in state:
                        logger.warning("[SLSK-DL] ❌ %s from %s: %s", short_fn, username, state)
                        return None
                    # Bail if stuck in queue too long
                    if "Queued" in state and pct == 0:
                        queued_polls += 1
                        if queued_polls >= 5:  # ~15s queued
                            logger.warning("[SLSK-DL] ⏭ Skipping %s — stuck in queue %ds", username, queued_polls * 3)
                            return None
                    else:
                        queued_polls = 0
                    # Track download start and bail if too slow
                    if "InProgress" in state:
                        if dl_start_time is None:
                            dl_start_time = _time.monotonic()
                        else:
                            dl_elapsed = _time.monotonic() - dl_start_time
                            # Only bail if projected total > 90s (very slow peer)
                            if dl_elapsed >= 15 and pct > 0 and pct < 15:
                                logger.warning("[SLSK-DL] ⏭ Skipping %s — too slow (%.1f%% in %.0fs)",
                                               username, pct, dl_elapsed)
                                return None
            if not found_file and poll > 5:
                logger.warning("[SLSK-DL] File not found in transfer list for %s", username)
                return None
    except Exception:
        logger.warning("Soulseek download failed for %s/%s", username, filename, exc_info=True)
    return None


def _find_downloaded_file(music_dir: str, filename: str) -> Path | None:
    """Search for a downloaded file. slskd always drops files into the root
    music directory, so also check parent dirs up to /music."""
    import time
    search_roots = [Path(music_dir)]
    # slskd downloads to the root music dir, not subdirs like _recommendations
    root = Path(os.getenv("MUSIC_DIR", "/music"))
    if root not in search_roots:
        search_roots.append(root)
    for music_path in search_roots:
        for p in music_path.rglob(filename):
            if time.time() - p.stat().st_mtime < 600:
                return p
        for p in music_path.rglob(f"*{Path(filename).suffix}"):
            if p.name == filename:
                return p
    return None


# ─── Batch Search ────────────────────────────────────────────────

async def search_batch(
    queries: list[dict],
    slskd_url: str = "",
    slskd_key: str = "",
    on_progress=None,
) -> list[UnifiedResult]:
    """Four-phase parallel batch search:
    1. Tidal (15 concurrent, verify Hi-Res)
    2. Retry not-found on Tidal with simplified query
    3. Soulseek for NOT-FOUND tracks
    4. Soulseek Hi-Res hunt for CD-only tracks
    """
    n = len(queries)
    results: list[UnifiedResult | None] = [None] * n
    done_count = 0

    # Phase 1: Tidal
    async with _proxy_client(timeout=15) as c:
        sem = asyncio.Semaphore(15)

        async def search_one(idx: int, q: dict):
            nonlocal done_count
            artist, title = q.get("artist", ""), q.get("title", "")
            query = _clean_query(artist, title)
            async with sem:
                tidal_results = await tidal_search(c, query, limit=10, verify_quality=True)
            results[idx] = UnifiedResult(
                query=query, artist=artist, title=title, sources=tidal_results,
            )
            done_count += 1
            if on_progress and (done_count % 10 == 0 or done_count == n):
                on_progress(done_count, n, "tidal")

        await asyncio.gather(
            *(search_one(i, q) for i, q in enumerate(queries)),
            return_exceptions=True,
        )

    for i in range(n):
        if results[i] is None:
            q = queries[i]
            results[i] = UnifiedResult(
                query=_clean_query(q.get("artist", ""), q.get("title", "")),
                artist=q.get("artist", ""), title=q.get("title", ""),
            )

    # Phase 2: Retry not-found on Tidal with title only
    not_found = [(i, results[i]) for i in range(n) if not results[i].best]
    if not_found:
        retry_count = 0
        async with _proxy_client(timeout=15) as c:
            sem2 = asyncio.Semaphore(15)

            async def retry_one(idx: int, r: UnifiedResult):
                nonlocal retry_count
                title_clean = re.sub(r'\s*[\(\[].*?[\)\]]', '', r.title)
                title_clean = re.sub(r'\s*(feat\.?|ft\.?).*', '', title_clean, flags=re.IGNORECASE)
                async with sem2:
                    retry_results = await tidal_search(c, title_clean, limit=5, verify_quality=False)
                r.sources.extend(retry_results)
                retry_count += 1
                if on_progress and (retry_count % 5 == 0 or retry_count == len(not_found)):
                    on_progress(retry_count, len(not_found), "retry")

            await asyncio.gather(
                *(retry_one(idx, r) for idx, r in not_found),
                return_exceptions=True,
            )

    # Phase 3 & 4: Soulseek
    if slskd_url and slskd_key:
        still_not_found = [(i, r) for i, r in enumerate(results) if not r.best]
        need_hires = [(i, r) for i, r in enumerate(results) if r.best and not r.best.is_hires]
        slsk_targets = still_not_found + need_hires

        if slsk_targets:
            slsk_done = 0
            async with httpx.AsyncClient(
                base_url=slskd_url.rstrip("/"),
                headers={"X-API-Key": slskd_key},
                timeout=30,
            ) as slsk_client:
                sem_slsk = asyncio.Semaphore(10)

                async def slsk_one(idx: int, r: UnifiedResult):
                    nonlocal slsk_done
                    async with sem_slsk:
                        slsk_res = await soulseek_search(slsk_client, r.query, wait=6)
                    r.sources.extend(slsk_res)
                    slsk_done += 1
                    if on_progress and (slsk_done % 10 == 0 or slsk_done == len(slsk_targets)):
                        on_progress(slsk_done, len(slsk_targets), "soulseek")

                await asyncio.gather(
                    *(slsk_one(idx, r) for idx, r in slsk_targets),
                    return_exceptions=True,
                )

    return results


# ─── Download with Fallback ──────────────────────────────────────

async def download_track(
    result: UnifiedResult,
    music_dir: str,
    slskd_url: str = "",
    slskd_key: str = "",
) -> DownloadResult | None:
    """Download best available version with fallback chain."""
    ranked = result.all_ranked
    if not ranked:
        return None

    Path(music_dir).mkdir(parents=True, exist_ok=True)
    orig_artist = result.artist
    orig_title = result.title

    # Dedup check
    check_artist = orig_artist or (ranked[0].artist if ranked else "")
    check_title = orig_title if orig_artist else (ranked[0].title if ranked else orig_title)
    existing = _find_existing(music_dir, check_artist, check_title)
    if existing:
        logger.info("[DEDUP] already exists — %s", existing)
        bd, sr = 16, 44100
        try:
            info = FLAC(existing).info
            bd = info.bits_per_sample or 16
            sr = info.sample_rate or 44100
        except Exception:
            pass
        return DownloadResult(
            path=existing, bit_depth=bd, sample_rate=sr,
            stream_type="existing", artist=check_artist, title=check_title,
        )

    use_source_meta = not orig_artist

    # SPEED-FIRST download: grab ANY lossless fast, upgrader handles hi-res later.
    # Monochrome API ONLY serves HI_RES_LOSSLESS (LOSSLESS/HIGH always 403).
    # Priority: Tidal HI_RES (instant) → Soulseek (FLAC first, smallest first)
    tidal_hires = [s for s in ranked if s.source == "tidal" and s.is_hires]
    # Separate FLAC from non-FLAC Soulseek results, sort each by size
    slsk_flac = sorted(
        [s for s in ranked if s.source == "soulseek" and s.filename and s.filename.lower().endswith(".flac")],
        key=lambda s: s.size_mb,
    )
    slsk_other = sorted(
        [s for s in ranked if s.source == "soulseek" and s.filename and not s.filename.lower().endswith(".flac")],
        key=lambda s: s.size_mb,
    )

    attempt_order: list[tuple[SourceResult, str | None]] = []
    # Tier 0: Tidal HI_RES — instant download, best quality
    for s in tidal_hires[:3]:
        attempt_order.append((s, "HI_RES_LOSSLESS"))
    # Tier 1: Soulseek FLAC first, then other lossless
    for s in slsk_flac[:5]:
        attempt_order.append((s, None))
    for s in slsk_other[:2]:
        attempt_order.append((s, None))

    tier_names = {0: "Tidal HI_RES", 1: "Soulseek"}
    tier = -1
    tidal_fails = 0
    for src, quality_override in attempt_order:
        new_tier = 0 if quality_override == "HI_RES_LOSSLESS" else 1
        if new_tier != tier:
            tier = new_tier
            logger.info("[DL-CHAIN] Tier %d: %s", tier, tier_names.get(tier, "?"))

        if src.source == "tidal" and tidal_fails >= 5:
            continue
        try:
            dl = await _download_source(
                src, music_dir, slskd_url, slskd_key,
                orig_artist=orig_artist, orig_title=orig_title,
                use_source_meta=use_source_meta,
                quality_override=quality_override,
            )
            if dl:
                return dl
            if src.source == "tidal":
                tidal_fails += 1
        except Exception:
            if src.source == "tidal":
                tidal_fails += 1
            continue
    return None


async def _download_source(
    src: SourceResult,
    music_dir: str,
    slskd_url: str,
    slskd_key: str,
    orig_artist: str = "",
    orig_title: str = "",
    use_source_meta: bool = False,
    quality_override: str | None = None,
) -> DownloadResult | None:
    if use_source_meta and src.artist:
        artist = src.artist
        title = src.title or orig_title
    else:
        artist = orig_artist or src.artist
        title = orig_title or src.title
    album = src.album

    if src.source == "tidal":
        quality = quality_override or ("HI_RES_LOSSLESS" if src.is_hires else "LOSSLESS")
        return await tidal_download(
            src.track_id, music_dir,
            artist=artist, title=title, album=album,
            quality=quality,
        )

    if src.source == "soulseek" and slskd_url and slskd_key and src.filename:
        async with httpx.AsyncClient(
            base_url=slskd_url.rstrip("/"),
            headers={"X-API-Key": slskd_key},
            timeout=90,
        ) as slsk_client:
            return await soulseek_download(
                slsk_client, src.peer, src.filename, music_dir,
                bit_depth=src.bit_depth, sample_rate=src.sample_rate,
                artist=artist, title=title,
                file_size=int(src.size_mb * 1_000_000),
                timeout=60,
            )
    return None


# ─── Library Upgrade Scanner ─────────────────────────────────────

async def upgrade_scan(
    music_dir: str,
    slskd_url: str,
    slskd_key: str,
    on_progress=None,
) -> list[dict]:
    """Scan tracks and search for quality upgrades.
    Checks: Tidal HI_RES_LOSSLESS first (fast), then Soulseek 24bit.
    Upgrades: any non-hires → hires, any lossy → lossless."""
    from mutagen.flac import FLAC as FLACFile

    upgrades = []
    upgradeable = []

    for f in Path(music_dir).rglob("*.flac"):
        if "_recommendations" in str(f):
            continue
        try:
            audio = FLACFile(str(f))
            bd = audio.info.bits_per_sample
            sr = audio.info.sample_rate
            if bd < 24:
                artist = (audio.get("artist") or [""])[0]
                title = (audio.get("title") or [""])[0]
                if artist and title:
                    upgradeable.append({
                        "path": str(f), "artist": artist, "title": title,
                        "bit_depth": bd, "sample_rate": sr,
                    })
        except Exception:
            continue

    if not upgradeable:
        return upgrades

    logger.info("[UPGRADE] Scan: %d tracks below 24-bit", len(upgradeable))

    # Phase 1: Check Tidal HI_RES_LOSSLESS (fast, parallel)
    async with _proxy_client(timeout=15) as c:
        sem = asyncio.Semaphore(10)

        async def check_tidal(track: dict) -> dict | None:
            query = _clean_query(track["artist"], track["title"])
            async with sem:
                results = await tidal_search(c, query, limit=5, verify_quality=True)
            for r in results:
                if not r.is_hires:
                    continue
                a_match = _fuzzy_match(r.artist, track["artist"])
                t_match = _fuzzy_match(r.title, track["title"])
                if a_match >= 0.3 and t_match >= 0.3:
                    if not _is_non_original(track["title"], r.title, r.version, r.album):
                        return {"track": track, "upgrade": r, "source": "tidal"}
            return None

        tidal_results = await asyncio.gather(
            *(check_tidal(t) for t in upgradeable), return_exceptions=True,
        )
        tidal_found = set()
        for res in tidal_results:
            if isinstance(res, dict) and res:
                upgrades.append(res)
                tidal_found.add(res["track"]["path"])
                logger.info("[UPGRADE] Tidal HI_RES: %s - %s",
                            res["track"]["artist"], res["track"]["title"])

    # Phase 2: Check Soulseek for tracks not found on Tidal
    remaining = [t for t in upgradeable if t["path"] not in tidal_found]
    if remaining and slskd_url and slskd_key:
        logger.info("[UPGRADE] Checking Soulseek for %d remaining tracks", len(remaining))
        async with httpx.AsyncClient(
            base_url=slskd_url.rstrip("/"),
            headers={"X-API-Key": slskd_key},
            timeout=30,
        ) as slsk_client:
            for i, track in enumerate(remaining):
                try:
                    query = f"{track['artist']} {track['title']}"
                    results = await soulseek_search(slsk_client, query, wait=10)

                    for r in results:
                        if r.bit_depth < 24:
                            continue
                        fname_lower = r.title.lower()
                        a_match = _fuzzy_match(fname_lower, track['artist'])
                        t_match = _fuzzy_match(fname_lower, track['title'])
                        if a_match < 0.3 or t_match < 0.3:
                            continue
                        if _is_non_original(track['title'], fname_lower):
                            continue
                        upgrades.append({"track": track, "upgrade": r, "source": "soulseek"})
                        logger.info("[UPGRADE] Soulseek 24bit: %s - %s  (%dkHz)",
                                    track["artist"], track["title"], r.sample_rate // 1000)
                        break

                    if on_progress:
                        on_progress(i + 1, len(remaining))
                    if (i + 1) % 10 == 0:
                        logger.info("[UPGRADE] Soulseek progress: %d/%d scanned",
                                    i + 1, len(remaining))
                except Exception:
                    continue
                await asyncio.sleep(2)

    logger.info("[UPGRADE] Scan done: %d candidates (%d Tidal, %d Soulseek)",
                len(upgrades),
                sum(1 for u in upgrades if u.get("source") == "tidal"),
                sum(1 for u in upgrades if u.get("source") == "soulseek"))
    return upgrades


async def upgrade_download(
    upgrade: dict,
    music_dir: str,
    slskd_url: str,
    slskd_key: str,
) -> DownloadResult | None:
    """Download upgrade from Tidal or Soulseek, verify quality, replace old file."""
    track = upgrade["track"]
    src = upgrade["upgrade"]
    old_path = track["path"]
    source_type = upgrade.get("source", "soulseek")

    dl = None
    if source_type == "tidal" and src.track_id:
        # Download from Tidal HI_RES
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="upgrade_")
        try:
            dl = await tidal_download(
                src.track_id, tmp_dir,
                artist=track["artist"], title=track["title"],
                quality="HI_RES_LOSSLESS",
            )
        except Exception:
            logger.warning("[UPGRADE] Tidal download failed for %s", track["title"], exc_info=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    elif src.filename and src.peer:
        # Download from Soulseek
        async with httpx.AsyncClient(
            base_url=slskd_url.rstrip("/"),
            headers={"X-API-Key": slskd_key},
            timeout=120,
        ) as slsk_client:
            dl = await soulseek_download(
                slsk_client, src.peer, src.filename, music_dir,
                bit_depth=src.bit_depth, sample_rate=src.sample_rate,
                artist=track["artist"], title=track["title"],
                file_size=int(src.size_mb * 1_000_000),
            )

    if dl and dl.path and Path(dl.path).exists():
        try:
            from mutagen.flac import FLAC as FLACVerify
            audio = FLACVerify(dl.path)
            actual_bd = audio.info.bits_per_sample
            if actual_bd < 24:
                logger.warning("[UPGRADE] %s - %s: claimed 24-bit but got %d-bit, keeping original",
                               track["artist"], track["title"], actual_bd)
                Path(dl.path).unlink(missing_ok=True)
                return None
        except Exception:
            logger.warning("[UPGRADE] %s - %s: couldn't verify quality",
                           track["artist"], track["title"])
            Path(dl.path).unlink(missing_ok=True)
            return None

        # Move upgraded file to correct location, remove old
        dest = _organize_path(music_dir, track["artist"], track["title"])
        if dl.path != dest:
            try:
                shutil.move(dl.path, dest)
                dl = DownloadResult(path=dest, bit_depth=dl.bit_depth,
                                    sample_rate=dl.sample_rate, stream_type=dl.stream_type,
                                    artist=dl.artist, title=dl.title, album=dl.album)
            except Exception:
                pass
        try:
            if old_path != dest:
                Path(old_path).unlink(missing_ok=True)
            parent = Path(old_path).parent
            if parent != Path(music_dir) and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass

        logger.info("[UPGRADE] ✅ Upgraded %s - %s: %dbit → %dbit",
                    track["artist"], track["title"], track["bit_depth"], actual_bd)
        return dl

    return None


# ─── Library Retag ───────────────────────────────────────────────

async def retag_library(music_dir: str, on_progress=None) -> dict:
    """Scan all FLAC files, find those missing metadata, fetch from Tidal + MusicBrainz and retag.

    Returns stats: {total, scanned, retagged, failed, skipped, details[]}.
    """
    music_path = Path(music_dir)
    flacs = sorted(music_path.rglob("*.flac"))
    total = len(flacs)
    retagged, failed, skipped = 0, 0, 0
    details = []

    for idx, fpath in enumerate(flacs):
        try:
            audio = FLAC(str(fpath))
            artist = (audio.get("artist") or [""])[0]
            title = (audio.get("title") or [""])[0]

            if not artist or not title:
                skipped += 1
                details.append({"file": fpath.name, "status": "skipped", "reason": "no artist/title"})
                continue

            # Check if already has full metadata (including genre)
            has_date = bool(audio.get("date"))
            has_tracknum = bool(audio.get("tracknumber"))
            has_isrc = bool(audio.get("isrc"))
            has_genre = bool(audio.get("genre"))

            if has_date and has_tracknum and has_isrc and has_genre:
                skipped += 1
                continue

            needs_tidal = not (has_date and has_tracknum and has_isrc)
            meta = {}

            if needs_tidal:
                # Search Tidal for this track
                query = _clean_query(artist, title)
                async with _proxy_client(timeout=15) as c:
                    results = await tidal_search(c, query, limit=5, verify_quality=False)

                if not results:
                    failed += 1
                    details.append({"file": fpath.name, "status": "not_found", "query": query})
                    # Still try genre from MusicBrainz
                    if not has_genre:
                        genre = await _fetch_genre_musicbrainz(artist)
                        if genre:
                            tag_flac(str(fpath), extra_meta={"genre": genre})
                            await asyncio.sleep(1.1)  # MusicBrainz rate limit: 1 req/s
                    continue

                # Find best matching result
                best = None
                for r in results:
                    a_match = _fuzzy_match(r.artist, artist)
                    t_match = _fuzzy_match(r.title, title)
                    if a_match >= 0.3 and t_match >= 0.5:
                        if not _is_non_original(title, r.title, r.version, r.album):
                            best = r
                            break
                if not best:
                    best = results[0]

                # Fetch full metadata
                meta = await _fetch_track_meta(best.track_id)
                if not meta:
                    failed += 1
                    details.append({"file": fpath.name, "status": "meta_fail", "track_id": best.track_id})
                    continue

            # Fetch genre from MusicBrainz if missing
            if not has_genre:
                genre = await _fetch_genre_musicbrainz(artist)
                if genre:
                    meta["genre"] = genre
                await asyncio.sleep(1.1)  # MusicBrainz rate limit

            # Apply metadata, keeping existing cover if present
            cover = meta.pop("cover_data", None)
            existing_cover = len(audio.pictures) > 0

            tag_flac(
                str(fpath),
                artist=artist,
                title=title,
                album=meta.get("album") or (audio.get("album") or [""])[0],
                cover_data=cover if not existing_cover else None,
                extra_meta=meta,
            )
            retagged += 1
            added = [k for k in ("date", "tracknumber", "isrc", "copyright", "discnumber", "genre")
                     if meta.get(k) and not audio.get(k.upper())]
            details.append({"file": fpath.name, "status": "retagged", "added": added})

        except Exception as e:
            failed += 1
            details.append({"file": fpath.name, "status": "error", "error": str(e)})

        if on_progress and ((idx + 1) % 5 == 0 or idx + 1 == total):
            on_progress(idx + 1, total, retagged)

    return {
        "total": total, "scanned": total,
        "retagged": retagged, "failed": failed, "skipped": skipped,
        "details": details,
    }
