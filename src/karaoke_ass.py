#!/usr/bin/env python3
"""
Build a karaoke-style .ass subtitle file, synced to real vocal timing,
fully configurable font/colors.

Reference mode:
  --lyrics given:   aligns your exact reference lyrics onto Whisper's word
                     timestamps via edit-distance alignment (handles Whisper
                     mishearing individual words -- still uses YOUR text).
  --lyrics omitted:  no-reference mode. Builds lines straight from Whisper's
                     own words, grouped on >0.6s gaps. Use this when there's
                     no transcript to work from -- run transcribe.py first.

Display mode:
  default:        one word on screen at a time -- appears when sung,
                   disappears right after (plus --buffer for held notes),
                   replaced by the next. No lingering full-line text.
  --line-karaoke: the older style -- the whole line stays on screen for its
                   duration, with each word tinted --highlight-color as it's
                   sung and --base-color before/after (classic sing-along
                   bar, not a disappearing-word style).

Colors accept #RRGGBB or a bare color name from a small built-in table.
"""
import argparse
import json
import re

WEB_COLORS = {
    "white": "#FFFFFF", "black": "#000000", "gold": "#FFD700", "yellow": "#FFFF00",
    "red": "#FF3B30", "orange": "#FF8C00", "cyan": "#00E5FF", "green": "#39FF14",
    "blue": "#3B82F6", "pink": "#FF3D9A", "purple": "#B026FF",
}


def hex_to_ass_bgr(color):
    color = WEB_COLORS.get(color.lower(), color)
    color = color.lstrip("#")
    r, g, b = color[0:2], color[2:4], color[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}"


def normalize(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def align_reference_lyrics(lyric_lines, whisper_words):
    filtered = [
        {"norm": normalize(w["word"]), "start": w["start"], "end": w["end"]}
        for w in whisper_words if normalize(w["word"])
    ]
    wnorm = [w["norm"] for w in filtered]

    lyric_words = []
    for i, line in enumerate(lyric_lines):
        for w in line.split():
            n = normalize(w)
            if n:
                lyric_words.append((i, n, w))
    lnorm = [w[1] for w in lyric_words]

    n, m = len(lnorm), len(wnorm)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if lnorm[i - 1] == wnorm[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    i, j, mapping = n, m, {}
    while i > 0 and j > 0:
        cost = 0 if lnorm[i - 1] == wnorm[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    # Fill gaps globally across the whole song, not per-line. Per-line
    # filling reset to prev_t=0.0 whenever a *line's first word* had no
    # Whisper match, which made unmatched opening words (ad-libs, shouted
    # intros Whisper mishears) flash on screen at t=0, before any singing.
    n_words = len(lyric_words)
    times = [filtered[mapping[idx]]["start"] if idx in mapping else None for idx in range(n_words)]
    ends = [filtered[mapping[idx]]["end"] if idx in mapping else None for idx in range(n_words)]
    for pos in range(n_words):
        if times[pos] is not None:
            continue
        prev_pos = max([p for p in range(pos) if times[p] is not None], default=None)
        next_pos = min([p for p in range(pos + 1, n_words) if times[p] is not None], default=None)
        if prev_pos is None and next_pos is None:
            prev_t, next_t, span, step = 0.0, 2.0, 1, 1
        elif prev_pos is None:
            # No matched word anywhere before this one -- anchor a short
            # fixed gap before the first known word instead of t=0, so
            # unmatched opening words don't pop in before the song starts.
            next_t = times[next_pos]
            prev_t = max(0.0, next_t - 0.4 * (next_pos - pos + 1))
            span, step = next_pos - pos + 1, 1
        elif next_pos is None:
            prev_t = times[prev_pos]
            next_t = prev_t + 2.0
            span, step = 2, pos - prev_pos
        else:
            prev_t, next_t = times[prev_pos], times[next_pos]
            span, step = next_pos - prev_pos, pos - prev_pos
        times[pos] = prev_t + (next_t - prev_t) * (step / max(span, 1))
        # No real Whisper word to take an end time from (this word was
        # skipped/mistranscribed) -- give it a short default duration
        # rather than leaving it unbounded.
        ends[pos] = times[pos] + 0.25

    words_by_line = {}
    for idx, (line_idx, _norm, original) in enumerate(lyric_words):
        words_by_line.setdefault(line_idx, []).append((idx, original))

    line_word_times = {}
    for line_idx, entries in words_by_line.items():
        line_word_times[line_idx] = [
            {"word": w, "start": times[idx], "end": ends[idx]} for idx, w in entries
        ]

    line_times = {li: v[0]["start"] for li, v in line_word_times.items()}
    return line_times, line_word_times


def no_reference_lines(whisper_words, gap_s=0.6):
    """Group Whisper's own words into lines on natural pauses -- no
    reference lyrics text needed at all."""
    lines, word_times = [], {}
    current, current_idx = [], 0
    prev_end = None
    for w in whisper_words:
        if not w["word"].strip():
            continue
        if prev_end is not None and w["start"] - prev_end > gap_s and current:
            lines.append(" ".join(x["word"] for x in current))
            word_times[current_idx] = current
            current_idx += 1
            current = []
        current.append(w)
        prev_end = w["end"]
    if current:
        lines.append(" ".join(x["word"] for x in current))
        word_times[current_idx] = current
    line_times = {i: v[0]["start"] for i, v in word_times.items()}
    return lines, line_times, word_times


def fmt(t):
    h, m = int(t // 3600), int((t % 3600) // 60)
    return f"{h:01d}:{m:02d}:{t % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words_json", help="Word-timestamp JSON from transcribe.py")
    ap.add_argument("duration", type=float, help="Total video duration in seconds")
    ap.add_argument("out", help="Output .ass path")
    ap.add_argument("--lyrics", help="Reference lyrics .txt (one line per subtitle cue). Omit for no-reference mode.")
    ap.add_argument("--font", default="Black Ops One", help="Font family name (must be installed, see fc-list)")
    ap.add_argument("--fontsize", type=int, default=88)
    ap.add_argument("--highlight-color", default="gold", help="Color words turn as they're sung (#RRGGBB or name)")
    ap.add_argument("--base-color", default="white", help="Color words start as before being sung")
    ap.add_argument("--outline-color", default="black")
    ap.add_argument("--outline-width", type=float, default=5)
    ap.add_argument("--caps", action="store_true", help="Force all text to UPPERCASE")
    ap.add_argument("--no-pop", action="store_true", help="Disable the entrance pop-in animation")
    ap.add_argument("--margin-v", type=int, default=130, help="Margin in px from the edge --align sits against")
    ap.add_argument("--align", default="bottom", choices=["top", "middle", "bottom"],
                     help="Vertical placement of the text block (default: bottom)")
    ap.add_argument("--line-karaoke", action="store_true",
                     help="Use the old style: whole line stays visible, words tint as sung (see module docstring)")
    ap.add_argument("--buffer", type=float, default=0.15,
                     help="Seconds a word lingers past its detected end before disappearing -- covers held notes (default: 0.15)")
    ap.add_argument("--min-display", type=float, default=0.12,
                     help="Minimum seconds a word stays visible even if sung very fast (default: 0.12)")
    args = ap.parse_args()

    with open(args.words_json) as f:
        whisper_words = json.load(f)

    if args.lyrics:
        with open(args.lyrics) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        line_times, line_word_times = align_reference_lyrics(lines, whisper_words)
    else:
        lines, line_times, line_word_times = no_reference_lines(whisper_words)

    align_num = {"top": 8, "middle": 5, "bottom": 2}[args.align]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,{args.font},{args.fontsize},{hex_to_ass_bgr(args.highlight_color)},{hex_to_ass_bgr(args.base_color)},{hex_to_ass_bgr(args.outline_color)},&H00000000,-1,0,0,0,100,100,2,0,1,{args.outline_width},3,{align_num},80,80,{args.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    pop = "" if args.no_pop else r"{\fscx45\fscy45\t(0,180,\fscx108\fscy108)\t(180,260,\fscx100\fscy100)}"
    rows = []
    sorted_idx = sorted(line_times.keys())

    if args.line_karaoke:
        # Old style: whole line stays on screen, words tint via \k as sung.
        for pos, idx in enumerate(sorted_idx):
            start = line_times[idx]
            end = line_times[sorted_idx[pos + 1]] if pos + 1 < len(sorted_idx) else args.duration
            words = line_word_times.get(idx, [])
            karaoke_text = ""
            for wpos, w in enumerate(words):
                w_start = w["start"]
                w_end = words[wpos + 1]["start"] if wpos + 1 < len(words) else end
                dur_cs = max(1, round((w_end - w_start) * 100))
                text = w["word"].upper() if args.caps else w["word"]
                karaoke_text += f"{{\\k{dur_cs}}}{text} "
            karaoke_text = karaoke_text.strip()
            rows.append(f"Dialogue: 0,{fmt(start)},{fmt(end)},Lyric,,0,0,0,,{pop}{karaoke_text}")
    else:
        # Default: one word cue at a time. Each word's own Dialogue entry
        # runs from its sung start to (its own end + buffer), but never past
        # the very next word's start (so two words can't overlap on screen)
        # and never shorter than --min-display (so fast runs don't flicker).
        flat = []
        for idx in sorted_idx:
            flat.extend(line_word_times.get(idx, []))
        flat.sort(key=lambda w: w["start"])

        for wpos, w in enumerate(flat):
            start = w["start"]
            next_start = flat[wpos + 1]["start"] if wpos + 1 < len(flat) else args.duration
            natural_end = w["end"] + args.buffer
            end = min(natural_end, next_start)
            end = max(end, start + args.min_display)
            end = min(end, args.duration)
            text = w["word"].upper() if args.caps else w["word"]
            rows.append(f"Dialogue: 0,{fmt(start)},{fmt(end)},Lyric,,0,0,0,,{pop}{text}")

    with open(args.out, "w") as f:
        f.write(header + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} karaoke cues -> {args.out}")


if __name__ == "__main__":
    main()
