# lyric-video-tools (CLI)

The pipeline behind the OG Karaoke Maker web app (`app/`) — usable
standalone if you'd rather script it than click through a form.

## Tools

- **transcribe.py** — faster-whisper word-level transcription. Pair with
  your own lyrics for accurate alignment, or use `--suggest-lyrics` to get a
  draft transcript when you don't have one (review it — Whisper mishears
  sung words more than spoken ones).
- **beat_detect.py** — librosa tempo/beat grid + onset ("hit") detection
  with per-hit strength, for driving particle effects and camera "boom"
  pulses off the actual music.
- **karaoke_ass.py** — builds a synced `.ass` subtitle file. Default style
  is one word on screen at a time — appears when sung, disappears right
  after (`--buffer` controls how long it lingers, for held notes) — pass
  `--line-karaoke` for the older style where the whole line stays up and
  words just tint as they're sung. Configurable font, size, position
  (`--align top|middle|bottom`), highlight/base color, caps, pop-in
  animation. Works with or without reference lyrics.
- **scene_compositor.py** — renders the cover-art camera motion (baseline
  zoom drift, handheld sway, beat-synced "boom" zoom-pulse) with the
  particle/smoke overlay screen-blended on top in one pass, in RGB (avoids
  the YUV chroma-blend pink-tint bug entirely). Every frame is independent,
  so it renders in parallel across all CPU cores by default (`--jobs` to
  override). Burst scatter area is configurable (`--burst-area-x0/x1/y0/y1`,
  0-1 fractions of width/height) — center-weighted by default, settable to
  full-screen.
- **particle_overlay.py** — older standalone particle-overlay tool
  (predates `scene_compositor.py`, superseded by it in the default
  pipeline; kept for reference / non-camera-motion use cases).
- **render_lyric_video.sh** — chains all of the above plus the ffmpeg
  compositing (frame-sequence encode, `.ass` burn-in, audio mux) into one
  command. Prints `STAGE: <name>` lines the web app parses for its progress
  bar.

## Setup

**Transcription/beat detection needs a GPU** (or patience) — this was built
against a `whisper-env` venv on a separate GPU box:

```
python3 -m venv ~/ai-companion/whisper-env
~/ai-companion/whisper-env/bin/pip install faster-whisper librosa
```

cuBLAS path quirk: if you hit `Library libcublas.so.12 is not found`, point
`LD_LIBRARY_PATH` at any other venv's bundled nvidia libs on the same box —
see the `LD_LIBRARY_PATH` line in `render_lyric_video.sh`.

Copy `transcribe.py` and `beat_detect.py` onto that GPU box at
`~/lyric-video-tools/` — `render_lyric_video.sh` scp's audio out and runs
them over ssh by default (`WHISPER_HOST` env var, defaults to `station`).
Set `TRANSCRIBE_LOCAL=1` to run them on the local machine instead if it has
its own GPU + the same packages installed.

**Font**: `karaoke_ass.py --font` must be an installed font (`fc-list`
checks). Download and drop into `~/.local/share/fonts/`, then `fc-cache -f`.

## Usage

```bash
# Full pipeline, with your own lyrics reference
./render_lyric_video.sh --song track.mp4 --cover art.png --out "Track Name" \
  --lyrics lyrics.txt --highlight-color gold --base-color white

# No lyrics available -- transcribe first, review the draft, then use it
ssh station '~/lyric-video-tools/transcribe.py audio.wav --out words.json --suggest-lyrics draft.txt'
# ...edit draft.txt to fix Whisper's mishearings...
./render_lyric_video.sh --song track.mp4 --cover art.png --out "Track Name" --lyrics draft.txt

# Turn up the particle chaos, scatter bursts across the whole frame
./render_lyric_video.sh --song track.mp4 --cover art.png --out "Track Name" \
  --burst-count-max 130 --burst-size-max 16 --burst-speed 600 \
  --burst-area-x0 0 --burst-area-x1 1 --burst-area-y0 0 --burst-area-y1 1

# Text centered mid-screen instead of the bottom, full-line karaoke style
./render_lyric_video.sh --song track.mp4 --cover art.png --out "Track Name" \
  --align middle --line-karaoke

# Smoke-only, no sparks, calmer mood
./render_lyric_video.sh --song track.mp4 --cover art.png --out "Track Name" --smoke-only
```

## Known gotchas (learned the hard way)

- **Never blend particle overlays in YUV** — `blend=screen` on yuv420p
  streams corrupts chroma and tints the whole frame pink. Always convert to
  `gbrp` (RGB) before blending, back to `yuv420p` only at final encode.
  `scene_compositor.py` handles this correctly by compositing entirely in
  RGB before the final encode step.
- **VAD filtering drops sung vocals** — `vad_filter=False` in transcribe.py
  is deliberate; the default VAD treats singing-over-instrumentation as
  non-speech far too often.
- **Full-length particle renders, not loops** — beat-synced bursts need
  absolute timing, so this renders the whole song, not a short seamless
  loop like a generic ambient-only overlay could.
- **`multiprocessing` needs an explicit `fork` context on some interpreters**
  — `scene_compositor.py` requests `multiprocessing.get_context("fork")`
  explicitly rather than relying on the interpreter's default start method.
  `spawn`/`forkserver` re-import the module fresh in each worker, which
  leaves the precomputed per-song particle/camera state empty there instead
  of inherited via copy-on-write.
- **An unmatched opening word can flash text at t=0** — if Whisper's
  transcript doesn't match a lyric line's very first word, gap-filling used
  to default to "no earlier match = t=0.0", making the line appear before
  any singing starts. `karaoke_ass.py`'s alignment now fills gaps globally
  across the whole song instead of resetting at each line boundary.
