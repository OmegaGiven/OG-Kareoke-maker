#!/usr/bin/env python3
"""
Render a beat-synced particle/smoke overlay as a full-length PNG sequence,
driven off beat_detect.py's output. Ambient particles drift continuously;
on each detected "hit" a radial burst fires, sized/counted by the hit's
strength and your multipliers below.

Output is a folder of frame_NNNNNN.png -- encode it yourself with e.g.:
  ffmpeg -framerate 24 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
then screen-blend it onto your background in RGB space (not YUV -- see
README.md for why that matters).
"""
import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
FPS = 24


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("beats_json", help="Output from beat_detect.py")
    ap.add_argument("duration", type=float, help="Total video duration in seconds")
    ap.add_argument("out_dir", help="Directory to write frame_NNNNNN.png into")
    ap.add_argument("--smoke-only", action="store_true", help="No ember/spark particles, just drifting smoke")
    ap.add_argument("--no-smoke", action="store_true", help="No smoke layer, just particles")
    ap.add_argument("--no-beat-pulse", action="store_true", help="Disable ambient brightness pulse on the beat grid")
    ap.add_argument("--ambient-count", type=int, default=70, help="Number of continuously-drifting background particles")
    ap.add_argument("--burst-size-min", type=float, default=2.0, help="Min spark radius at burst peak (px)")
    ap.add_argument("--burst-size-max", type=float, default=5.5, help="Max spark radius at burst peak (px)")
    ap.add_argument("--burst-count-min", type=int, default=6, help="Sparks per hit at strength=0")
    ap.add_argument("--burst-count-max", type=int, default=28, help="Sparks per hit at strength=1 (scales linearly between)")
    ap.add_argument("--burst-speed", type=float, default=270, help="Base outward spark speed (px/s), scaled by hit strength")
    ap.add_argument("--burst-lifetime", type=float, default=0.55, help="Seconds a burst's sparks stay visible")
    ap.add_argument("--smoke-count", type=int, default=8)
    ap.add_argument("--color", default="255,150,40", help="Ember/spark R,G,B at full brightness")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    with open(args.beats_json) as f:
        beats = json.load(f)
    beat_times = beats["beat_times"]
    hits = beats["hits"]
    spark_color = tuple(int(c) for c in args.color.split(","))

    random.seed(args.seed)
    frames = int(args.duration * FPS) + 1

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
            cx, cy = random.uniform(W * 0.2, W * 0.8), random.uniform(H * 0.3, H * 0.75)
            sparks = [{
                "ang": random.uniform(0, 6.28),
                "speed": random.uniform(args.burst_speed * 0.45, args.burst_speed * 1.55) * (0.5 + strength),
                "size": random.uniform(args.burst_size_min, args.burst_size_max) * (0.6 + strength),
            } for _ in range(n_sparks)]
            bursts.append({"time": h["time"], "cx": cx, "cy": cy, "sparks": sparks})
        bursts.sort(key=lambda b: b["time"])

    def nearest_beat_dist(t):
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

    os.makedirs(args.out_dir, exist_ok=True)
    active_start_idx = 0

    for f in range(frames):
        t = f / FPS
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
        img = Image.merge("RGB", (smoke_layer, smoke_layer, smoke_layer))
        draw = ImageDraw.Draw(img)

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

        while active_start_idx < len(bursts) and bursts[active_start_idx]["time"] + args.burst_lifetime < t:
            active_start_idx += 1
        for b in bursts[active_start_idx:]:
            if b["time"] > t:
                break
            age = t - b["time"]
            if age > args.burst_lifetime:
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

        img = img.filter(ImageFilter.GaussianBlur(0.6))
        img.save(f"{args.out_dir}/frame_{f:06d}.png")

    print(f"frames written: {frames}")


if __name__ == "__main__":
    main()
