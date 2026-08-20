#!/usr/bin/env python3
"""
Detect tempo/beat grid and percussive/orchestral onset "hits" in a song,
for driving beat-synced particle effects. Needs librosa (CPU-only, no GPU
required -- runs fine anywhere Python is set up for it).
"""
import argparse
import json

import librosa
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="Path to audio file (wav/mp3/anything librosa can load)")
    ap.add_argument("--out", required=True, help="Output beats/hits JSON path")
    ap.add_argument("--delta", type=float, default=0.12,
                     help="Onset sensitivity, lower = more hits detected (default: 0.12)")
    ap.add_argument("--wait", type=int, default=3,
                     help="Min frames between detected onsets, higher = fewer/sparser hits (default: 3)")
    args = ap.parse_args()

    y, sr = librosa.load(args.audio, sr=22050, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=False, delta=args.delta, wait=args.wait
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    strengths = onset_env[onset_frames] if len(onset_frames) else np.array([])
    if len(strengths):
        strengths = ((strengths - strengths.min()) / (strengths.max() - strengths.min() + 1e-9)).tolist()
    else:
        strengths = []

    tempo_val = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)
    with open(args.out, "w") as f:
        json.dump({
            "tempo": tempo_val,
            "beat_times": beat_times,
            "hits": [{"time": t, "strength": s} for t, s in zip(onset_times, strengths)],
        }, f, indent=2)

    print(f"tempo={tempo_val:.1f}bpm, {len(beat_times)} beats, {len(onset_times)} hits -> {args.out}")


if __name__ == "__main__":
    main()
