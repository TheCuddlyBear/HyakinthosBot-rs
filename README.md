# Rust Discord Music Bot (Lavalink v4)

A Discord music bot built in Rust using [`poise`](https://github.com/serenity-rs/poise), [`serenity`](https://github.com/serenity-rs/serenity), [`songbird`](https://github.com/serenity-rs/songbird), and [`lavalink-rs`](https://gitlab.com/vicky5124/lavalink-rs).

## Commands

All commands work as both **Slash Commands** (`/`) and **Prefix Commands** (default `!`):

| Command | Description |
| :--- | :--- |
| `/play [query]` | Plays a track or playlist from a URL or YouTube search query. Automatically joins your voice channel. |
| `/play-local <file>` | Accepts an audio/video file attachment uploaded to Discord and plays it through Lavalink. |
| `/skip` | Skips the currently playing track and advances the queue. |
| `/queue` | Displays the currently playing track and the upcoming songs in the queue. |
| `/stop` | Stops playback and clears the current queue. |
| `/volume <level>` | Sets the player volume (`0` to `1000`, default `100`). |
| `/pause [state]` | Pauses or resumes the current track (toggles if `state` is omitted). |

## Setup

### 1. Lavalink v4 Server
Ensure you have a **Lavalink v4** server running.
- You can use the provided [`application.yml.example`](file:///c:/Users/basvt/Desktop/rust%20discord%20music/application.yml.example) as a starting configuration.
- Make sure `http: true` and `local: true` are enabled under `lavalink.server.sources` in your `application.yml` so `/play-local` can stream Discord attachment URLs and local fallback files.

### 2. Environment Variables
Copy [`.env.example`](file:///c:/Users/basvt/Desktop/rust%20discord%20music/.env.example) to `.env` and fill in your values:

```env
DISCORD_TOKEN=your_discord_bot_token_here
LAVALINK_HOST=localhost:2333
LAVALINK_PASSWORD=youshallnotpass
LAVALINK_SSL=false
BOT_PREFIX=!
```

> **Note:** Make sure **Message Content Intent** and **Server Members / Voice States** are enabled in the Discord Developer Portal if you also want to use prefix commands (`!play`, `!play-local` with attachments, etc.).

### 3. Run the Bot
```powershell
cargo run --release
```
