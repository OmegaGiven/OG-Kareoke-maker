#!/usr/bin/env python3
"""
Local web app wrapping the lyric-video-tools CLI pipeline: upload a song +
cover art (+ optional lyrics), tune font/position/particle/camera settings,
preview the particle effect on a short clip, then render the full video with
a live progress bar. All the actual rendering work happens in
render_lyric_video.sh / scene_compositor.py -- this is an orchestration
layer, not a reimplementation.

Run:
  pip install -r requirements.txt
  uvicorn server:app --host 0.0.0.0 --port 8091 --app-dir app

Env vars (all optional):
  WHISPER_HOST     ssh alias of a GPU box with whisper-env set up (default: station)
  TRANSCRIBE_LOCAL set to 1 to run transcribe.py/beat_detect.py on this machine instead
  JOBS_DIR         where uploads/renders live (default: ./data/jobs next to this file)
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

TOOLS_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(os.environ.get("JOBS_DIR", Path(__file__).resolve().parent / "data" / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OG Karaoke Maker")

# job_id -> {stage, percent, log (list[str]), done, error, output_path}
JOBS = {}
JOBS_LOCK = threading.Lock()

FRAME_TOTAL_RE = re.compile(r"rendering (\d+) frames")
FRAME_PROGRESS_RE = re.compile(r"^\s*(\d+)/(\d+)\s*$")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")


def clean_lyrics_text(text):
    """Strip bracketed stage directions ([chorus], [verse 2], ad-libs, etc.)
    line by line, dropping any line left empty after stripping."""
    lines = []
    for raw in text.splitlines():
        line = BRACKETED_RE.sub("", raw).strip()
        if line:
            lines.append(line)
    return lines

STAGE_WEIGHTS = {
    "extract_audio": 2,
    "transcribe": 25,
    "karaoke": 3,
    "scene": 55,
    "encode": 8,
    "mux": 5,
    "done": 2,
}
STAGE_ORDER = list(STAGE_WEIGHTS.keys())


def _stage_base_percent(stage):
    idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    return sum(STAGE_WEIGHTS[s] for s in STAGE_ORDER[:idx])


def _run_job(job_id, cmd, env, cwd):
    job = JOBS[job_id]
    frames_total = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            with JOBS_LOCK:
                job["log"].append(line)
                job["log"] = job["log"][-200:]
                if line.startswith("STAGE: "):
                    stage = line[len("STAGE: "):].strip()
                    job["stage"] = stage
                    job["percent"] = _stage_base_percent(stage)
                else:
                    m = FRAME_TOTAL_RE.search(line)
                    if m:
                        frames_total = int(m.group(1))
                    m = FRAME_PROGRESS_RE.match(line)
                    if m and frames_total:
                        done, total = int(m.group(1)), int(m.group(2))
                        frac = min(1.0, done / total) if total else 0.0
                        job["percent"] = _stage_base_percent("scene") + frac * STAGE_WEIGHTS["scene"]
        proc.wait()
        with JOBS_LOCK:
            if proc.returncode == 0:
                job["percent"] = 100
                job["done"] = True
            else:
                job["error"] = f"render exited with code {proc.returncode}"
                job["done"] = True
    except Exception as e:
        with JOBS_LOCK:
            job["error"] = str(e)
            job["done"] = True


@app.post("/api/jobs")
async def create_job(
    song: UploadFile = File(...),
    cover: UploadFile = File(...),
    lyrics: UploadFile = File(None),
    lyrics_text: str = Form(""),
    out_name: str = Form("My Song"),
    font: str = Form("Black Ops One"),
    fontsize: int = Form(88),
    align: str = Form("bottom"),
    margin_v: int = Form(130),
    highlight_color: str = Form("gold"),
    base_color: str = Form("white"),
    caps: bool = Form(True),
    line_karaoke: bool = Form(False),
    buffer_s: float = Form(0.15),
    min_display: float = Form(0.12),
    smoke_only: bool = Form(False),
    no_smoke: bool = Form(False),
    no_beat_pulse: bool = Form(False),
    ambient_count: int = Form(70),
    burst_count_max: int = Form(28),
    burst_size_max: float = Form(5.5),
    burst_speed: float = Form(270),
    burst_area_x0: float = Form(0.2),
    burst_area_x1: float = Form(0.8),
    burst_area_y0: float = Form(0.3),
    burst_area_y1: float = Form(0.75),
    zoom_speed: float = Form(0.0006),
    sway_x: float = Form(40),
    sway_y: float = Form(25),
    sway_freq_x: float = Form(40),
    sway_freq_y: float = Form(55),
    onset_delta: float = Form(0.12),
    boom_amplitude: float = Form(0.10),
    boom_attack: float = Form(0.05),
    boom_decay: float = Form(0.22),
    boom_min_strength: float = Form(0.0),
    no_boom: bool = Form(False),
    jobs: int = Form(0),
    particle_color: str = Form("255,150,40"),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    song_path = job_dir / f"song{Path(song.filename).suffix}"
    cover_path = job_dir / f"cover{Path(cover.filename).suffix}"
    with open(song_path, "wb") as f:
        shutil.copyfileobj(song.file, f)
    with open(cover_path, "wb") as f:
        shutil.copyfileobj(cover.file, f)

    # Text box takes priority over an uploaded file if both are given;
    # bracketed stage directions ([chorus], [verse 2], ad-libs, ...) are
    # stripped either way -- karaoke_ass.py should only ever see sung lines.
    raw_lyrics = None
    if lyrics_text.strip():
        raw_lyrics = lyrics_text
    elif lyrics is not None and lyrics.filename:
        raw_lyrics = (await lyrics.read()).decode("utf-8", errors="replace")

    lyrics_path = None
    if raw_lyrics is not None:
        cleaned = clean_lyrics_text(raw_lyrics)
        if cleaned:
            lyrics_path = job_dir / "lyrics.txt"
            with open(lyrics_path, "w") as f:
                f.write("\n".join(cleaned) + "\n")

    output_path = job_dir / f"{out_name} - Lyric Video.mp4"

    cmd = [
        str(TOOLS_DIR / "render_lyric_video.sh"),
        "--song", str(song_path), "--cover", str(cover_path), "--out", out_name,
        "--output-path", str(output_path),
        "--font", font, "--fontsize", str(fontsize),
        "--align", align, "--margin-v", str(margin_v),
        "--highlight-color", highlight_color, "--base-color", base_color,
        "--buffer", str(buffer_s), "--min-display", str(min_display),
        "--ambient-count", str(ambient_count),
        "--burst-count-max", str(burst_count_max), "--burst-size-max", str(burst_size_max),
        "--burst-speed", str(burst_speed),
        "--burst-area-x0", str(burst_area_x0), "--burst-area-x1", str(burst_area_x1),
        "--burst-area-y0", str(burst_area_y0), "--burst-area-y1", str(burst_area_y1),
        "--zoom-speed", str(zoom_speed), "--sway-x", str(sway_x), "--sway-y", str(sway_y),
        "--sway-freq-x", str(sway_freq_x), "--sway-freq-y", str(sway_freq_y),
        "--onset-delta", str(onset_delta),
        "--boom-amplitude", str(boom_amplitude), "--boom-attack", str(boom_attack),
        "--boom-decay", str(boom_decay), "--boom-min-strength", str(boom_min_strength),
        "--jobs", str(jobs),
        "--particle-color", particle_color,
    ]
    if lyrics_path:
        cmd += ["--lyrics", str(lyrics_path)]
    if not caps:
        cmd += ["--no-caps"]
    if line_karaoke:
        cmd += ["--line-karaoke"]
    if smoke_only:
        cmd += ["--smoke-only"]
    if no_smoke:
        cmd += ["--no-smoke"]
    if no_beat_pulse:
        cmd += ["--no-beat-pulse"]
    if no_boom:
        cmd += ["--no-boom"]

    env = dict(os.environ)
    env.setdefault("WHISPER_HOST", os.environ.get("WHISPER_HOST", "station"))
    env.setdefault("TRANSCRIBE_LOCAL", os.environ.get("TRANSCRIBE_LOCAL", "0"))
    env["PYTHONUNBUFFERED"] = "1"

    JOBS[job_id] = {
        "stage": "queued", "percent": 0, "log": [], "done": False,
        "error": None, "output_path": str(output_path),
    }
    t = threading.Thread(target=_run_job, args=(job_id, cmd, env, str(TOOLS_DIR)), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    with JOBS_LOCK:
        return {
            "stage": job["stage"], "percent": round(job["percent"], 1),
            "done": job["done"], "error": job["error"],
            "log_tail": job["log"][-15:],
            "ready": job["done"] and not job["error"],
        }


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job["done"] or job["error"]:
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(job["output_path"], filename=Path(job["output_path"]).name)


PREVIEW_DUMMY_LINES = ["YOUR LYRICS HERE", "PREVIEW LOOKS LIKE THIS"]
PREVIEW_DUMMY_WORDS = [w for line in PREVIEW_DUMMY_LINES for w in line.split()]


@app.post("/api/preview")
async def preview(
    cover: UploadFile = File(...),
    # particles
    ambient_count: int = Form(70),
    burst_count_max: int = Form(28),
    burst_size_max: float = Form(5.5),
    burst_speed: float = Form(270),
    burst_area_x0: float = Form(0.2),
    burst_area_x1: float = Form(0.8),
    burst_area_y0: float = Form(0.3),
    burst_area_y1: float = Form(0.75),
    smoke_only: bool = Form(False),
    no_smoke: bool = Form(False),
    particle_color: str = Form("255,150,40"),
    # camera
    zoom_speed: float = Form(0.0006),
    sway_x: float = Form(40),
    sway_y: float = Form(25),
    sway_freq_x: float = Form(40),
    sway_freq_y: float = Form(55),
    boom_amplitude: float = Form(0.10),
    boom_attack: float = Form(0.05),
    boom_decay: float = Form(0.22),
    no_boom: bool = Form(False),
    burst_now: bool = Form(False),
    # text
    show_text: bool = Form(True),
    font: str = Form("Black Ops One"),
    fontsize: int = Form(88),
    align: str = Form("bottom"),
    margin_v: int = Form(130),
    highlight_color: str = Form("gold"),
    base_color: str = Form("white"),
    caps: bool = Form(True),
    line_karaoke: bool = Form(False),
    lyrics_text: str = Form(""),
):
    """Renders a short (4s), looping synthetic clip so the UI can show
    camera motion + particles + karaoke text style together, at current
    settings, without waiting on a full song render. Runs the exact same
    scene_compositor.py / karaoke_ass.py code paths as a real render --
    this is WYSIWYG, not a separate mock renderer. Shows the user's actual
    (bracket-stripped) lyrics if any have been entered yet, else a generic
    placeholder line so text styling is still visible before lyrics exist."""
    job_id = "preview-" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    cover_path = job_dir / f"cover{Path(cover.filename).suffix}"
    with open(cover_path, "wb") as f:
        shutil.copyfileobj(cover.file, f)

    duration = 4.0
    # Evenly spaced fake "hits" so bursts + boom pulses are visibly exercised.
    hits = [{"time": t, "strength": 0.6 + 0.4 * (i % 2)} for i, t in enumerate([0.4, 1.2, 2.0, 2.8, 3.5])]
    if burst_now:
        # "Trigger particle burst" -- force one unmissable max-strength hit
        # right at the start instead of relying on the user to notice one
        # of the regular spaced-out hits mid-loop.
        hits = [{"time": 0.15, "strength": 1.0}] + hits
    beats = {"beat_times": [i * 0.5 for i in range(int(duration / 0.5))], "hits": hits}
    beats_path = job_dir / "beats.json"
    with open(beats_path, "w") as f:
        json.dump(beats, f)

    cleaned = clean_lyrics_text(lyrics_text) if lyrics_text.strip() else []
    preview_lines = cleaned[:2] if cleaned else PREVIEW_DUMMY_LINES
    preview_words = [w for line in preview_lines for w in line.split()][:10] or PREVIEW_DUMMY_WORDS

    # Word timings spaced evenly across the clip, spelled identically to
    # preview_lines so karaoke_ass.py's alignment matches every word exactly
    # -- no interpolation guesswork needed for a preview.
    span = duration - 0.6
    step = span / max(len(preview_words), 1)
    words_json = []
    for i, w in enumerate(preview_words):
        start = 0.3 + i * step
        words_json.append({"word": w, "start": start, "end": start + min(step * 0.7, 0.35)})
    words_path = job_dir / "words.json"
    with open(words_path, "w") as f:
        json.dump(words_json, f)
    lyrics_path = job_dir / "lyrics.txt"
    with open(lyrics_path, "w") as f:
        f.write("\n".join(preview_lines) + "\n")

    ass_path = job_dir / "lyrics.ass"
    ass_cmd = [
        "python3", str(TOOLS_DIR / "karaoke_ass.py"), str(words_path), str(duration), str(ass_path),
        "--lyrics", str(lyrics_path), "--font", font, "--fontsize", str(fontsize),
        "--align", align, "--margin-v", str(margin_v),
        "--highlight-color", highlight_color, "--base-color", base_color,
    ]
    if caps:
        ass_cmd.append("--caps")
    if line_karaoke:
        ass_cmd.append("--line-karaoke")
    subprocess.run(ass_cmd, check=True, cwd=str(TOOLS_DIR))

    frames_dir = job_dir / "frames"
    camera_json_path = job_dir / "camera.json"
    scene_cmd = [
        "python3", str(TOOLS_DIR / "scene_compositor.py"),
        str(cover_path), str(beats_path), str(duration), str(frames_dir),
        "--ambient-count", str(ambient_count),
        "--burst-count-max", str(burst_count_max), "--burst-size-max", str(burst_size_max),
        "--burst-speed", str(burst_speed),
        "--burst-area-x0", str(burst_area_x0), "--burst-area-x1", str(burst_area_x1),
        "--burst-area-y0", str(burst_area_y0), "--burst-area-y1", str(burst_area_y1),
        "--color", particle_color,
        "--zoom-drift", str(zoom_speed), "--sway-x", str(sway_x), "--sway-y", str(sway_y),
        "--sway-freq-x", str(sway_freq_x), "--sway-freq-y", str(sway_freq_y),
        "--boom-amplitude", str(boom_amplitude), "--boom-attack", str(boom_attack),
        "--boom-decay", str(boom_decay),
        "--camera-json", str(camera_json_path),
    ]
    if smoke_only:
        scene_cmd.append("--smoke-only")
    if no_smoke:
        scene_cmd.append("--no-smoke")
    if no_boom:
        scene_cmd.append("--no-boom")
    subprocess.run(scene_cmd, check=True, cwd=str(TOOLS_DIR))

    visual_path = job_dir / "visual.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(visual_path), "-loglevel", "error",
    ], check=True)

    out_path = job_dir / "preview.mp4"
    if show_text:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(visual_path),
            "-filter_complex", f"[0:v]ass={ass_path}[outv]", "-map", "[outv]",
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            str(out_path), "-loglevel", "error",
        ], check=True)
    else:
        visual_path.rename(out_path)

    shutil.rmtree(frames_dir, ignore_errors=True)
    return FileResponse(out_path, media_type="video/mp4", headers={"X-Preview-Job-Id": job_id})


@app.get("/api/preview/{job_id}/camera-path")
def preview_camera_path(job_id: str):
    """Per-frame crop-window rects (fraction of the full cover image) for
    the given preview job, so the UI can show a moving camera-crop minimap
    synced to video playback instead of a static frame guide."""
    path = JOBS_DIR / job_id / "camera.json"
    if not path.exists():
        return JSONResponse({"error": "unknown preview job"}, status_code=404)
    return FileResponse(path, media_type="application/json")


@app.get("/api/fonts")
def list_fonts():
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, check=True).stdout
        families = sorted({f.split(",")[0].strip() for f in out.splitlines() if f.strip()})
        return {"fonts": families}
    except Exception:
        return {"fonts": ["Black Ops One", "Bebas Neue", "Arial"]}


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="static")
