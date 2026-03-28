# 🎵 musomatic-server

Self-hosted lossless music API server. Searches Tidal + Soulseek in parallel, downloads FLAC in original quality, auto-upgrades 16→24bit, and generates AI-powered recommendations.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   📱 iPhone Shortcuts        💻 CLI Client                  │
│   "Shazam → Download"       $ musomatic download ...       │
│         │                          │                        │
│         └──────────┬───────────────┘                        │
│                    ▼                                        │
│         ┌──────────────────┐                                │
│         │  Musomatic API   │  FastAPI server                │
│         │  search+download │  Background 16→24bit upgrader  │
│         │  quality check   │  🤖 AI recommendations         │
│         └───────┬──────────┘                                │
│           ┌─────┴─────┐                                     │
│           ▼           ▼                                     │
│     ┌──────────┐ ┌──────────┐                               │
│     │  Tidal   │ │ Soulseek │  Multi-source search          │
│     │(lossless)│ │  (slskd) │  Hi-Res 24bit preferred       │
│     └──────────┘ └────┬─────┘                               │
│                       │                                     │
│              ┌────────▼────────┐                             │
│              │   /music (FLAC) │  Organized library          │
│              │  Artist/Track   │  Artist/Artist - Track.flac │
│              └────────┬────────┘                             │
│                       │                                     │
│              ┌────────▼────────┐                             │
│              │   Navidrome     │  Subsonic streaming server  │
│              │   Web UI + API  │  Auto-scans /music          │
│              └────────┬────────┘                             │
│                       │                                     │
│         ┌─────────────┼─────────────┐                       │
│         ▼             ▼             ▼                        │
│     📱 Amperfy    🌐 Web UI    🖥 Feishin                   │
│     (iOS+cache)  (Navidrome)  (Desktop)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-source search** — Tidal (Monochrome API) + Soulseek (slskd) in parallel
- **Hi-Res preferred** — 24bit/96-192kHz when available, CD quality fallback
- **Smart matching** — popularity scoring, fuzzy matching, artist+title verification
- **Non-original filtering** — auto-skips remixes, live, covers, karaoke, demos
- **Background upgrader** — replaces 16-bit with 24-bit from Soulseek every 4h
- **🤖 AI recommendations** — LLM analyzes your library, suggests new tracks, auto-downloads into a playlist
- **Dedup detection** — won't re-download what you already have
- **FLAC tagging** — artist, title, album, cover art
- **Batch downloads** — hundreds of tracks with progress tracking
- **iPhone Shortcuts** — Shazam → download in one tap
- **API key auth** — LAN trusted, external requires key

## Quick Start

```bash
git clone https://github.com/9ckobn/musomatic-server.git
cd musomatic-server
cp .env.example .env
# Edit .env with your settings
docker compose up -d
```

### Required `.env` settings

```bash
MUSIC_DIR=/path/to/your/music
SLSKD_USERNAME=your_soulseek_login
SLSKD_PASSWORD=your_soulseek_password
SLSKD_KEY=$(openssl rand -base64 32)
```

### Containers

| Container | Port | Description |
|-----------|------|-------------|
| `music-api` | 8844 | Musomatic API |
| `slskd` | 5030 | Soulseek client |
| `navidrome` | 4533 | Streaming server |

## 🤖 AI Recommendations

Musomatic can analyze your library with an LLM and generate personalized playlists.

### Setup

Add to `.env`:
```bash
# Pick a provider: openai, deepseek, claude, openrouter
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-your-key-here

# Navidrome credentials (for playlist creation)
NAVIDROME_USER=admin
NAVIDROME_PASSWORD=your-password

# Optional: auto-generate every N seconds (0 = manual only)
RECOMMEND_INTERVAL=0
# Delete unrated tracks after N hours
RECOMMEND_CLEANUP_HOURS=24
```

### How it works

1. **Analyze** — LLM reads your library (artists + titles) and suggests ~30 tracks you'd like
2. **Download** — tracks are searched on Tidal/Soulseek and downloaded to `_recommendations/`
3. **Playlist** — "AI Recommendations" playlist appears in Navidrome/Amperfy
4. **Rate** — star or rate tracks you enjoy → they move to your main library
5. **Cleanup** — after 24h, unrated tracks are auto-deleted

### Supported LLM providers

| Provider | Model (default) | Cost |
|----------|----------------|------|
| `deepseek` | deepseek-chat | ~$0.01/run |
| `openai` | gpt-4o-mini | ~$0.02/run |
| `claude` | claude-sonnet-4-20250514 | ~$0.03/run |
| `openrouter` | deepseek/deepseek-chat | varies |

### CLI usage

```bash
# Generate recommendations (uses .env settings)
musomatic recommend

# Custom provider + count
musomatic recommend -p openai -k sk-xxx -n 20

# Check status
musomatic recommend status

# Manual cleanup (keep rated, delete rest)
musomatic recommend cleanup
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status |
| POST | `/search` | Search tracks |
| POST | `/download` | Download single (async job) |
| POST | `/quick` | Sync search + download |
| POST | `/batch/download` | Batch download (async job) |
| GET | `/library/tracks` | List library (`?q=search`) |
| GET | `/library/stats` | Library statistics |
| GET | `/library/audit` | Audit for non-originals |
| POST | `/library/delete` | Delete tracks |
| POST | `/upgrade/trigger` | Manual upgrade scan |
| GET | `/upgrade/status` | Upgrade status |
| POST | `/recommend/generate` | Generate AI recommendations |
| GET | `/recommend/status` | Recommendation status |
| POST | `/recommend/cleanup` | Clean up unrated recommendations |
| GET | `/jobs` | List jobs |
| GET | `/jobs/{id}` | Job status |
| POST | `/jobs/{id}/cancel` | Cancel job |

## External Access

Behind CGNAT? Use **Cloudflare Tunnel**:

```bash
cloudflared tunnel create home-server
cloudflared tunnel route dns home-server music.yourdomain.com
cloudflared tunnel route dns home-server api.yourdomain.com
```

See [docs/security.md](docs/security.md) for full setup.

## Client

Install the CLI client from [musomatic-client](https://github.com/9ckobn/musomatic-client).

## License

MIT
