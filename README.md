# OG Karaoke Maker

Self-hosted lyric/karaoke video pipeline: audio → word-level transcription →
lyric alignment → karaoke `.ass` subtitles → beat detection → beat-synced
particle effects + camera motion → composited video. No third-party paid
services — runs on your own machine (GPU recommended for transcription).

Two ways to use it:

- **Web app** (`src/app/`) — upload a song + cover art, tune font/position,
  particle intensity/scatter, camera sway/zoom/boom, preview the particle
  effect on a short clip, render the full video with a live progress bar.
- **CLI** (`src/render_lyric_video.sh`) — same pipeline, scriptable, good
  for batch-processing an album.

## Quick start (web app)

```bash
cd src/app
pip install -r requirements.txt
uvicorn server:app --app-dir . --host 0.0.0.0 --port 8091
```

Open `http://localhost:8091`. See `src/app/README.md` for env vars
(`WHISPER_HOST`, `TRANSCRIBE_LOCAL`, `JOBS_DIR`).

## Quick start (CLI)

```bash
cd src
./render_lyric_video.sh --song track.wav --cover art.png --out "Track Name" \
  --lyrics lyrics.txt --highlight-color gold --base-color white
```

Run with no args' worth of flags to see the full option list at the top of
the script, or read `src/README.md`.

## Pipeline tools (`src/`)

| Tool | Purpose |
|---|---|
| `transcribe.py` | faster-whisper word-level transcription |
| `beat_detect.py` | librosa tempo/beat grid + onset ("hit") detection |
| `karaoke_ass.py` | builds synced `.ass` karaoke subtitles (word-at-a-time or full-line, configurable position/font/color) |
| `scene_compositor.py` | renders cover-art camera motion (zoom/sway/beat-synced boom) + particle/smoke overlay, parallelized across CPU cores |
| `particle_overlay.py` | older standalone particle overlay (superseded by `scene_compositor.py` in the default pipeline, kept for reference) |
| `render_lyric_video.sh` | chains all of the above + ffmpeg muxing into one command |
| `app/` | FastAPI + vanilla-JS web UI wrapping the pipeline |

## Setup notes

**Transcription needs a GPU** (or patience). See `src/README.md` for the
`whisper-env` venv setup and the `LD_LIBRARY_PATH` cuBLAS workaround some
boxes need.

**Fonts**: `karaoke_ass.py --font` must be an installed font (`fc-list`
checks it). Drop `.ttf`/`.otf` into `~/.local/share/fonts/`, then
`fc-cache -f`.

## Known gotchas

- **Never blend particle overlays in YUV** — screen-blend on `yuv420p`
  corrupts chroma and tints the frame pink. Always blend in RGB (`gbrp`),
  convert to `yuv420p` only at final encode. `scene_compositor.py` already
  does this correctly.
- **VAD filtering drops sung vocals** — `transcribe.py` always passes
  `vad_filter=False`; the default VAD treats singing-over-instrumentation
  as non-speech far too often.
- **Full-length particle renders, not loops** — beat-synced bursts need
  absolute timing against the song, so rendering is per-song, not a
  seamless loop.
- **`multiprocessing` needs an explicit `fork` context** — `scene_compositor.py`
  requests `multiprocessing.get_context("fork")` explicitly; the interpreter's
  default start method isn't guaranteed to be `fork`, and `spawn`/`forkserver`
  re-import the module fresh per worker instead of inheriting the parent's
  precomputed particle/camera schedules.
