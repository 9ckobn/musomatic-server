# 📱 iPhone Shortcuts Setup

Musomatic integrates with iOS via two Shortcuts: **quick download** and **Shazam-to-download**.

## Prerequisites

- Musomatic server accessible from iPhone (via LAN or Cloudflare Tunnel / VPN)
- Server URL (e.g., `http://192.168.1.100:8844` or `https://api.yourdomain.com`)
- API key (if using external access)

---

## Shortcut 1: "Download Track" 🎵

Downloads a track by typing "Artist - Title".

### Setup

1. Open **Shortcuts** app on iPhone
2. Create new Shortcut → **Add Action**
3. Add: **Ask for Input**
   - Question: `What track?`
   - Input Type: **Text**
4. Add: **Get Contents of URL**
   - URL: `http://YOUR_SERVER:8844/quick`
   - Method: **POST**
   - Headers: `Content-Type: application/json` (and `x-api-key: YOUR_KEY` if needed)
   - Request Body: JSON → `{"query": "Provided Input"}`
5. Add: **Get Dictionary Value**
   - Key: `message`
6. Add: **Show Notification**
   - Use the dictionary value as notification text

### Usage

Run the shortcut → type `Rammstein - Du Hast` → get notification with quality info.

---

## Shortcut 2: "Shazam Download" 🎤

Shazam a song playing nearby and auto-download it.

### Setup

1. Create new Shortcut → **Add Action**
2. Add: **Shazam It** (from Music Recognition)
3. Add: **Get Details of Shazam**
   - Get: **Title**
   - Save to variable `shazam_title`
4. Add: **Get Details of Shazam**
   - Get: **Artist**
   - Save to variable `shazam_artist`
5. Add: **Text**
   - Content: `[shazam_artist] - [shazam_title]`
6. Add: **Get Contents of URL**
   - URL: `http://YOUR_SERVER:8844/quick`
   - Method: **POST**
   - Headers: `Content-Type: application/json` (and `x-api-key: YOUR_KEY` if needed)
   - Request Body: JSON → `{"query": "Text"}`
7. Add: **Get Dictionary Value**
   - Key: `message`
8. Add: **Show Notification**

### Usage

Run shortcut → hold phone near speaker → Shazam identifies → auto-downloads in lossless.

---

## Tips

- Add shortcuts to Home Screen for quick access
- Use **Automation** to trigger Shazam shortcut with a button tap
- If using Cloudflare Tunnel, use `https://api.yourdomain.com/quick` as the URL
- API key goes in the `x-api-key` header
