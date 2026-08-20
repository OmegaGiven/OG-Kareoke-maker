#!/usr/bin/env python3
"""
Renders the full visual background for a lyric video in one pass: cover art
with slow baseline zoom-drift + handheld sway + beat-synced "camera boom"
push-in/out pulses, with the particle/smoke overlay screen-blended directly
on top in RGB (no separate ffmpeg blend step, and no risk of the YUV
chroma-blend pink-tint bug from doing this in ffmpeg's zoompan+blend chain).

Every frame is independent to compute, so this renders in parallel across
CPU cores (multiprocessing, default: all of them -- see --jobs) instead of
one frame at a time on a single core. On Linux this costs nothing extra in
setup: worker processes are forked *after* all the heavy per-song data
(cover art pixels, particle/burst schedules) is built once in the parent,
so each child inherits it via copy-on-write instead of re-building or
re-serializing it per frame.

Output is a folder of frame_NNNNNN.png -- encode with e.g.:
  ffmpeg -framerate 24 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
then mux audio + burn karaoke .ass on top of that (no more blend needed).
"""
import argparse
import bisect
import json
import math
import multiprocessing
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
FPS = 24

# Populated once in main() before the worker pool is forked -- every worker
# inherits these via copy-on-write (Linux fork), so there's no per-frame or
# per-worker re-serialization cost even though cover_arr can be tens of MB.
G = {}


def screen_blend(bg, fg):
    """Standard screen blend, vectorized: 255 - (255-a)*(255-b)/255."""
    bg_f = bg.astype(np.float32)
    fg_f = fg.astype(np.float32)
    return (255 - (255 - bg_f) * (255 - fg_f) / 255).astype(np.uint8)


def nearest_beat_dist(t):
    beat_times = G["beat_times"]
    if not beat_times:
        return 999
    lo, hi = 0, len(beat_times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if beat_times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    cand = [beat_times[max(lo - 1, 0)], beat_times[lo]]
    return min(abs(t - c) for c in cand)


def boom_zoom_extra(t):
    """Fast-attack/slow-decay punch centered on the nearest hit within
    range -- not a sum over all hits (keeps it a punch, not a blur).
    Deliberately a plain linear scan over G['hits'] -- unchanged from the
    single-core version on purpose (a dense hit list is a wanted creative
    choice here, not something to shortcut); parallelism is what makes a
    dense hit list affordable, not thinning the hits themselves."""
    args = G["args"]
    hits = G["hits"]
    if args.no_boom or not hits:
        return 0.0
    best = None
    for h in hits:
        dt = t - h["time"]
        if -0.02 <= dt <= args.boom_attack + args.boom_decay * 3:
            if best is None or abs(dt) < abs(best[0]):
                best = (dt, h["strength"])
    if best is None:
        return 0.0
    dt, strength = best
    if dt < args.boom_attack:
        frac = max(0.0, dt) / max(args.boom_attack, 1e-6)
        env = frac
    else:
        env = math.exp(-(dt - args.boom_attack) / max(args.boom_decay, 1e-6))
    return args.boom_amplitude * strength * env


def render_frame(f):
    args = G["args"]
    cover_arr = G["cover_arr"]
    big_h, big_w = G["big_h"], G["big_w"]
    ambient = G["ambient"]
    smoke_plumes = G["smoke_plumes"]
    bursts = G["bursts"]
    burst_times = G["burst_times"]
    spark_color = G["spark_color"]

    t = f / FPS

    # --- background: zoom/sway/boom ---
    drift = min(args.zoom_drift * f, args.zoom_drift_max)
    boom = boom_zoom_extra(t)
    zoom = 1.0 + drift + boom

    crop_w = min(big_w, W / zoom)
    crop_h = min(big_h, H / zoom)
    sway_x = args.sway_x * math.sin(f / args.sway_freq_x)
    sway_y = args.sway_y * math.cos(f / args.sway_freq_y)
    cx = big_w / 2 + sway_x - crop_w / 2
    cy = big_h / 2 + sway_y - crop_h / 2
    cx = max(0, min(big_w - crop_w, cx))
    cy = max(0, min(big_h - crop_h, cy))
    crop = cover_arr[int(cy):int(cy + crop_h), int(cx):int(cx + crop_w)]
    bg_img = Image.fromarray(crop).resize((W, H), Image.LANCZOS)
    bg_arr = np.array(bg_img)

    # --- foreground: particles/smoke ---
    smoke_layer = Image.new("L", (W, H), 0)
    if smoke_plumes:
        sdraw = ImageDraw.Draw(smoke_layer)
        for s in smoke_plumes:
            x = (s["x"] + s["speed_x"] * t) % (W + 400) - 200
            y = s["y"] + s["speed_y"] * t
            wob = 20 * ((t * 0.4 + s["phase"]) % 6.28 - 3.14) / 3.14
            r = s["r"]
            sdraw.ellipse([x - r + wob, y - r * 0.6, x + r + wob, y + r * 0.6], fill=s["opacity"])
        smoke_layer = smoke_layer.filter(ImageFilter.GaussianBlur(60))
    fg_img = Image.merge("RGB", (smoke_layer, smoke_layer, smoke_layer))
    draw = ImageDraw.Draw(fg_img)

    if ambient:
        pulse = 0.0
        if not args.no_beat_pulse:
            pulse = max(0.0, 1.0 - nearest_beat_dist(t) / 0.25)
        for p in ambient:
            x = (p["x"] + p["speed_x"] * t) % W
            y = (p["y"] + p["speed_y"] * t) % H
            sway = 10 * (0.5 + 0.5 * ((t * 1.3 + p["phase"]) % 6.28 / 6.28))
            x = (x + sway) % W
            boost = 1.0 + 0.5 * pulse
            color = (spark_color[0], min(255, int(spark_color[1] * 0.55 * boost)), spark_color[2]) if p["glow"] else (110, 100, 95)
            r = p["size"] * (1.0 + 0.3 * pulse)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Frames now render out of order across worker processes, so the old
    # single-core version's incremental "active_start_idx" pointer (which
    # relied on strictly-increasing t between calls) can't be reused --
    # bisect the sorted burst-times list for the active window instead.
    # Still O(log n) to find it, same as the sequential pointer was
    # amortized to, just correct under out-of-order execution.
    if bursts:
        lo = bisect.bisect_left(burst_times, t - args.burst_lifetime)
        hi = bisect.bisect_right(burst_times, t)
        for b in bursts[lo:hi]:
            age = t - b["time"]
            if age > args.burst_lifetime or age < 0:
                continue
            life = 1.0 - (age / args.burst_lifetime)
            for sp in b["sparks"]:
                dist = sp["speed"] * age
                sx = b["cx"] + dist * math.cos(sp["ang"])
                sy = b["cy"] + dist * math.sin(sp["ang"])
                r = sp["size"] * life
                if r <= 0.3:
                    continue
                c = (spark_color[0], int(spark_color[1] * life + 40), int(spark_color[2] * life))
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=c)

    fg_img = fg_img.filter(ImageFilter.GaussianBlur(0.6))
    fg_arr = np.array(fg_img)

    composited = screen_blend(bg_arr, fg_arr)
    Image.fromarray(composited).save(f"{G['out_dir']}/frame_{f:06d}.png")
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cover", help="Cover art image path")
    ap.add_argument("beats_json", help="Output from beat_detect.py")
    ap.add_argument("duration", type=float)
    ap.add_argument("out_dir", help="Directory to write frame_NNNNNN.png into")

    ap.add_argument("--zoom-drift", type=float, default=0.0006, help="Per-frame baseline zoom-in creep")
    ap.add_argument("--zoom-drift-max", type=float, default=0.15)
    ap.add_argument("--sway-x", type=float, default=40)
    ap.add_argument("--sway-y", type=float, default=25)
    ap.add_argument("--sway-freq-x", type=float, default=40, help="Lower = faster/shakier")
    ap.add_argument("--sway-freq-y", type=float, default=55)

    ap.add_argument("--boom-amplitude", type=float, default=0.10, help="Extra zoom fraction at the peak of a boom")
    ap.add_argument("--boom-attack", type=float, default=0.05, help="Seconds to punch in to peak zoom")
    ap.add_argument("--boom-decay", type=float, default=0.22, help="Seconds to settle back out")
    ap.add_argument("--boom-min-strength", type=float, default=0.0, help="Only boom on hits at/above this strength (0-1)")
    ap.add_argument("--no-boom", action="store_true", help="Disable beat-synced boom, baseline zoom/sway only")

    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--no-beat-pulse", action="store_true")
    ap.add_argument("--ambient-count", type=int, default=70)
    ap.add_argument("--burst-size-min", type=float, default=2.0)
    ap.add_argument("--burst-size-max", type=float, default=5.5)
    ap.add_argument("--burst-count-min", type=int, default=6)
    ap.add_argument("--burst-count-max", type=int, default=28)
    ap.add_argument("--burst-speed", type=float, default=270)
    ap.add_argument("--burst-lifetime", type=float, default=0.55)
    ap.add_argument("--burst-area-x0", type=float, default=0.2, help="Left edge of where burst centers can spawn, 0-1 fraction of width")
    ap.add_argument("--burst-area-x1", type=float, default=0.8, help="Right edge, 0-1 fraction of width")
    ap.add_argument("--burst-area-y0", type=float, default=0.3, help="Top edge, 0-1 fraction of height")
    ap.add_argument("--burst-area-y1", type=float, default=0.75, help="Bottom edge, 0-1 fraction of height")
    ap.add_argument("--smoke-count", type=int, default=8)
    ap.add_argument("--color", default="255,150,40")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--jobs", type=int, default=0, help="Worker processes (default: 0 = all CPU cores)")
    ap.add_argument("--camera-json", help="Optional: dump per-frame crop-window rects (fraction of the full cover "
                     "image) to this path, e.g. for a UI to show the moving camera crop against the full art")
    args = ap.parse_args()

    with open(args.beats_json) as f:
        beats = json.load(f)
    beat_times = beats["beat_times"]
    hits = [h for h in beats["hits"] if h["strength"] >= args.boom_min_strength]
    spark_color = tuple(int(c) for c in args.color.split(","))

    random.seed(args.seed)
    frames = int(args.duration * FPS) + 1

    max_zoom = 1.0 + args.zoom_drift_max + (0 if args.no_boom else args.boom_amplitude) + 0.05
    cover = Image.open(args.cover).convert("RGB")
    cw, ch = cover.size
    target_ratio = W / H
    if cw / ch > target_ratio:
        new_h = int(H * max_zoom * 1.3)
        new_w = int(new_h * cw / ch)
    else:
        new_w = int(W * max_zoom * 1.3)
        new_h = int(new_w * ch / cw)
    cover_big = cover.resize((new_w, new_h), Image.LANCZOS)
    cover_arr = np.array(cover_big)
    big_h, big_w = cover_arr.shape[:2]

    ambient = []
    if not args.smoke_only:
        for _ in range(args.ambient_count):
            ambient.append({
                "x": random.uniform(0, W), "y": random.uniform(0, H),
                "size": random.uniform(1.5, 4.0),
                "speed_y": random.uniform(-16, -6), "speed_x": random.uniform(-6, 6),
                "phase": random.uniform(0, 6.28),
                "glow": random.choice([True, True, False]),
            })
    smoke_plumes = []
    if not args.no_smoke:
        for _ in range(args.smoke_count):
            smoke_plumes.append({
                "x": random.uniform(-200, W + 200), "y": random.uniform(H * 0.25, H * 1.1),
                "r": random.uniform(200, 380), "speed_x": random.uniform(2, 8),
                "speed_y": random.uniform(-3, -0.5), "phase": random.uniform(0, 6.28),
                "opacity": random.randint(16, 28),
            })
    bursts = []
    if not args.smoke_only:
        for h in hits:
            strength = h["strength"]
            n_sparks = int(args.burst_count_min + strength * (args.burst_count_max - args.burst_count_min))
            cx = random.uniform(W * args.burst_area_x0, W * args.burst_area_x1)
            cy = random.uniform(H * args.burst_area_y0, H * args.burst_area_y1)
            sparks = [{
                "ang": random.uniform(0, 6.28),
                "speed": random.uniform(args.burst_speed * 0.45, args.burst_speed * 1.55) * (0.5 + strength),
                "size": random.uniform(args.burst_size_min, args.burst_size_max) * (0.6 + strength),
            } for _ in range(n_sparks)]
            bursts.append({"time": h["time"], "cx": cx, "cy": cy, "sparks": sparks})
        bursts.sort(key=lambda b: b["time"])

    os.makedirs(args.out_dir, exist_ok=True)

    G.update({
        "args": args, "cover_arr": cover_arr, "big_h": big_h, "big_w": big_w,
        "beat_times": beat_times, "hits": hits, "ambient": ambient,
        "smoke_plumes": smoke_plumes, "bursts": bursts,
        "burst_times": [b["time"] for b in bursts],
        "spark_color": spark_color, "out_dir": args.out_dir,
    })

    if args.camera_json:
        # Camera transform math only (no pixel work), so this is cheap to
        # run single-threaded even for a full song -- lets a UI show the
        # moving crop window against the full cover art without decoding
        # rendered frames itself.
        samples = []
        for f in range(frames):
            t = f / FPS
            drift = min(args.zoom_drift * f, args.zoom_drift_max)
            boom = boom_zoom_extra(t)  # reads G["args"]/G["hits"], already populated above
            zoom = 1.0 + drift + boom
            crop_w = min(big_w, W / zoom)
            crop_h = min(big_h, H / zoom)
            sway_x = args.sway_x * math.sin(f / args.sway_freq_x)
            sway_y = args.sway_y * math.cos(f / args.sway_freq_y)
            cx = big_w / 2 + sway_x - crop_w / 2
            cy = big_h / 2 + sway_y - crop_h / 2
            cx = max(0, min(big_w - crop_w, cx))
            cy = max(0, min(big_h - crop_h, cy))
            samples.append({
                "t": round(t, 3),
                "x0": round(cx / big_w, 4), "x1": round((cx + crop_w) / big_w, 4),
                "y0": round(cy / big_h, 4), "y1": round((cy + crop_h) / big_h, 4),
            })
        with open(args.camera_json, "w") as f:
            json.dump(samples, f)

    n_jobs = args.jobs if args.jobs > 0 else os.cpu_count()
    print(f"rendering {frames} frames across {n_jobs} worker processes...")
    # Explicit 'fork' context, not whatever multiprocessing defaults to on
    # this Python version -- G's contents (cover_arr etc, populated just
    # above) only reach workers for free via fork's copy-on-write semantics.
    # 'spawn'/'forkserver' re-import the module fresh in each worker instead,
    # which leaves G empty there (observed directly: KeyError on G["args"]
    # in every worker under whatever this interpreter's real default is).
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_jobs) as pool:
        done = 0
        for _ in pool.imap_unordered(render_frame, range(frames), chunksize=16):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{frames}")

    print(f"frames written: {frames}")


if __name__ == "__main__":
    main()
