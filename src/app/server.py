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

    lyrics_path = None
    if lyrics is not None and lyrics.filename:
        lyrics_path = job_dir / "lyrics.txt"
        with open(lyrics_path, "wb") as f:
            shutil.copyfileobj(lyrics.file, f)

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


@app.post("/api/preview/particles")
async def preview_particles(
    cover: UploadFile = File(...),
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
    zoom_speed: float = Form(0.0006),
    sway_x: float = Form(40),
    sway_y: float = Form(25),
    sway_freq_x: float = Form(40),
    sway_freq_y: float = Form(55),
    boom_amplitude: float = Form(0.10),
    boom_attack: float = Form(0.05),
    boom_decay: float = Form(0.22),
    no_boom: bool = Form(False),
):
    """Renders a short (4s) synthetic clip so the UI can show the particle/
    camera settings in action without waiting on a full song render."""
    job_id = "preview-" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    cover_path = job_dir / f"cover{Path(cover.filename).suffix}"
    with open(cover_path, "wb") as f:
        shutil.copyfileobj(cover.file, f)

    duration = 4.0
    # Evenly spaced fake "hits" so bursts + boom pulses are visibly exercised.
    beats = {
        "beat_times": [i * 0.5 for i in range(int(duration / 0.5))],
        "hits": [{"time": t, "strength": 0.6 + 0.4 * (i % 2)} for i, t in enumerate([0.4, 1.2, 2.0, 2.8, 3.5])],
    }
    beats_path = job_dir / "beats.json"
    with open(beats_path, "w") as f:
        json.dump(beats, f)

    frames_dir = job_dir / "frames"
    cmd = [
        "python3", str(TOOLS_DIR / "scene_compositor.py"),
        str(cover_path), str(beats_path), str(duration), str(frames_dir),
        "--ambient-count", str(ambient_count),
        "--burst-count-max", str(burst_count_max), "--burst-size-max", str(burst_size_max),
        "--burst-speed", str(burst_speed),
        "--burst-area-x0", str(burst_area_x0), "--burst-area-x1", str(burst_area_x1),
        "--burst-area-y0", str(burst_area_y0), "--burst-area-y1", str(burst_area_y1),
        "--zoom-drift", str(zoom_speed), "--sway-x", str(sway_x), "--sway-y", str(sway_y),
        "--sway-freq-x", str(sway_freq_x), "--sway-freq-y", str(sway_freq_y),
        "--boom-amplitude", str(boom_amplitude), "--boom-attack", str(boom_attack),
        "--boom-decay", str(boom_decay),
    ]
    if smoke_only:
        cmd.append("--smoke-only")
    if no_smoke:
        cmd.append("--no-smoke")
    if no_boom:
        cmd.append("--no-boom")

    subprocess.run(cmd, check=True, cwd=str(TOOLS_DIR))
    out_path = job_dir / "preview.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(out_path), "-loglevel", "error",
    ], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    return FileResponse(out_path, media_type="video/mp4")


@app.get("/api/fonts")
def list_fonts():
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, check=True).stdout
        families = sorted({f.split(",")[0].strip() for f in out.splitlines() if f.strip()})
        return {"fonts": families}
    except Exception:
        return {"fonts": ["Black Ops One", "Bebas Neue", "Arial"]}


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="static")
