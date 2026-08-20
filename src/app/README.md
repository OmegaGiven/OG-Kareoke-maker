# OG Karaoke Maker — web app

A small local web app wrapping the CLI toolkit in `../` (transcribe, beat
detection, karaoke `.ass` generation, particle/camera compositing) with a
form UI: upload song + cover, tune font/position/particles/camera, preview
the particle effect on a short clip, then render the full video with a
progress bar.

This is an orchestration layer only — it shells out to
`render_lyric_video.sh` and `scene_compositor.py`, it doesn't reimplement
any rendering logic.

## Deployed instance

Runs persistently on `station` as a `systemctl --user` service
(`~/.config/systemd/user/og-karaoke-maker.service`), port 8094, using the
`~/ai-companion/whisper-env` venv (already has faster-whisper/librosa for
transcription; fastapi/uvicorn/pillow were added to it so one venv covers
the whole pipeline — see that service file's `Environment=` lines for the
cuBLAS `LD_LIBRARY_PATH` workaround and `TRANSCRIBE_LOCAL=1`). Listed on
go's Homepage dashboard under AI as "OG Karaoke Maker".

```bash
ssh station systemctl --user status og-karaoke-maker
ssh station journalctl --user -u og-karaoke-maker -f
```

## Run (manual / dev)

```bash
pip install -r requirements.txt
uvicorn server:app --app-dir . --host 0.0.0.0 --port 8091
```

Open `http://localhost:8091`.

## Env vars

- `WHISPER_HOST` — ssh alias of a GPU box with `~/ai-companion/whisper-env`
  set up (see `../README.md`). Default: `station`.
- `TRANSCRIBE_LOCAL=1` — run transcription/beat-detection on this machine
  instead of over ssh (needs the same packages installed locally).
- `JOBS_DIR` — where uploads and renders are stored. Default:
  `./data/jobs` next to `server.py`.

## What's exposed vs. what's hardcoded

Font, text position (top/middle/bottom) + size, colors, caps toggle,
word-at-a-time vs. full-line karaoke, particle counts/sizes/speed, burst
scatter area (center-weighted vs. full-screen), camera sway/zoom/boom, and
onset sensitivity are all in the UI. Anything not exposed (e.g. individual
color-per-word gradients, non-rectangular burst zones) is still available
as a flag on the underlying CLI tools if you need it — the app just doesn't
have a control for it yet.

## Progress bar

`render_lyric_video.sh` prints `STAGE: <name>` lines at each pipeline step
plus frame-count checkpoints during the compositing stage; the backend
parses both from the subprocess's stdout to estimate percent complete.
It's a weighted estimate (transcription and compositing dominate the
total time), not a byte-exact measurement.
