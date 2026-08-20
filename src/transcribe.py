#!/usr/bin/env python3
"""
Transcribe a song's vocals to word-level timestamps using faster-whisper.
Two uses:
  1. Reference alignment: pair this output with your own lyrics.txt in
     karaoke_ass.py so it maps your exact words onto the sung timing.
  2. No-lyrics fallback: pass --suggest-lyrics to also dump a lyrics.txt
     built straight from what Whisper heard (line breaks inserted on >0.6s
     gaps) -- a starting draft you review/correct before using it as the
     reference lyrics file, since Whisper mishears words on sung audio.

Must run in an environment with faster-whisper + a GPU with working cuBLAS
(see README.md in this folder for the exact station setup this was built
against). VAD is off by default -- it drops sung vocals as "non-speech"
far more often than it helps.
"""
import argparse
import json

from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="Path to a mono/stereo wav (or anything ffmpeg can decode)")
    ap.add_argument("--out", required=True, help="Output path for word-timestamp JSON")
    ap.add_argument("--model", default="medium.en", help="faster-whisper model name (default: medium.en)")
    ap.add_argument("--vad", action="store_true", help="Enable VAD filtering (usually hurts on sung audio -- off by default)")
    ap.add_argument("--suggest-lyrics", help="Also write a draft lyrics.txt here, line-broken on >0.6s gaps")
    args = ap.parse_args()

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    segments, _info = model.transcribe(args.audio, word_timestamps=True, vad_filter=args.vad)

    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    with open(args.out, "w") as f:
        json.dump(words, f, indent=2)
    print(f"transcribed {len(words)} words -> {args.out}")

    if args.suggest_lyrics:
        lines, current = [], []
        prev_end = None
        for w in words:
            if not w["word"]:
                continue
            if prev_end is not None and w["start"] - prev_end > 0.6 and current:
                lines.append(" ".join(current))
                current = []
            current.append(w["word"])
            prev_end = w["end"]
        if current:
            lines.append(" ".join(current))
        with open(args.suggest_lyrics, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"draft lyrics ({len(lines)} lines) -> {args.suggest_lyrics} -- review before using as reference")


if __name__ == "__main__":
    main()
