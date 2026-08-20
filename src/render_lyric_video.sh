#!/bin/bash
# Master pipeline: song (audio, or a video with an audio track) -> lyric
# video with karaoke text, beat-synced particles, camera sway, and optional
# beat-synced "boom" zoom pulses. See README.md for setup (whisper-env on a
# GPU box, librosa, ffmpeg locally).
#
# Usage:
#   ./render_lyric_video.sh --song track.wav --cover art.png --out "My Song" \
#     [--lyrics lyrics.txt]            # omit for no-reference transcription mode
#     [--font "Bebas Neue"] [--fontsize 88]
#     [--highlight-color gold] [--base-color white]
#     [--smoke-only] [--no-smoke] [--no-beat-pulse]
#     [--burst-count-max 28] [--burst-size-max 5.5] [--burst-speed 270]
#     [--zoom-speed 0.0006] [--sway-x 40] [--sway-y 25]
#     [--sway-freq-x 40] [--sway-freq-y 55]   # LOWER = faster/shakier wobble
#     [--onset-delta 0.12]                    # LOWER = more particle hits detected
#     [--boom-amplitude 0.10] [--boom-attack 0.05] [--boom-decay 0.22]
#     [--boom-min-strength 0.0] [--no-boom]   # beat-synced push-zoom pulse
#
# Requires: WHISPER_HOST (ssh alias of a machine with a working GPU +
# ~/ai-companion/whisper-env set up per README.md), or set TRANSCRIBE_LOCAL=1
# to run transcribe.py/beat_detect.py on this machine instead.

set -e
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SONG="" COVER="" OUT="" LYRICS="" FONT="Black Ops One" FONTSIZE=88
HIGHLIGHT="gold" BASE="white" SMOKE_FLAG="" NO_SMOKE_FLAG="" NO_PULSE_FLAG=""
BURST_COUNT_MAX=28 BURST_SIZE_MAX=5.5 BURST_SPEED=270
BURST_AREA_X0=0.2 BURST_AREA_X1=0.8 BURST_AREA_Y0=0.3 BURST_AREA_Y1=0.75
PARTICLE_COLOR="255,150,40"
ZOOM_SPEED=0.0006 SWAY_X=40 SWAY_Y=25 SWAY_FREQ_X=40 SWAY_FREQ_Y=55
ONSET_DELTA=0.12
BOOM_AMPLITUDE=0.10 BOOM_ATTACK=0.05 BOOM_DECAY=0.22 BOOM_MIN_STRENGTH=0.0 NO_BOOM_FLAG=""
JOBS=0
ALIGN="bottom" MARGIN_V=130 CAPS_FLAG="--caps" LINE_KARAOKE_FLAG="" BUFFER=0.15 MIN_DISPLAY=0.12
OUTPUT_PATH=""
WHISPER_HOST="${WHISPER_HOST:-station}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --song) SONG="$2"; shift 2 ;;
    --cover) COVER="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --lyrics) LYRICS="$2"; shift 2 ;;
    --font) FONT="$2"; shift 2 ;;
    --fontsize) FONTSIZE="$2"; shift 2 ;;
    --highlight-color) HIGHLIGHT="$2"; shift 2 ;;
    --base-color) BASE="$2"; shift 2 ;;
    --smoke-only) SMOKE_FLAG="--smoke-only"; shift ;;
    --no-smoke) NO_SMOKE_FLAG="--no-smoke"; shift ;;
    --no-beat-pulse) NO_PULSE_FLAG="--no-beat-pulse"; shift ;;
    --burst-count-max) BURST_COUNT_MAX="$2"; shift 2 ;;
    --burst-size-max) BURST_SIZE_MAX="$2"; shift 2 ;;
    --burst-speed) BURST_SPEED="$2"; shift 2 ;;
    --burst-area-x0) BURST_AREA_X0="$2"; shift 2 ;;
    --burst-area-x1) BURST_AREA_X1="$2"; shift 2 ;;
    --burst-area-y0) BURST_AREA_Y0="$2"; shift 2 ;;
    --burst-area-y1) BURST_AREA_Y1="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --align) ALIGN="$2"; shift 2 ;;
    --margin-v) MARGIN_V="$2"; shift 2 ;;
    --no-caps) CAPS_FLAG=""; shift ;;
    --line-karaoke) LINE_KARAOKE_FLAG="--line-karaoke"; shift ;;
    --buffer) BUFFER="$2"; shift 2 ;;
    --min-display) MIN_DISPLAY="$2"; shift 2 ;;
    --particle-color) PARTICLE_COLOR="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --zoom-speed) ZOOM_SPEED="$2"; shift 2 ;;
    --sway-x) SWAY_X="$2"; shift 2 ;;
    --sway-y) SWAY_Y="$2"; shift 2 ;;
    --sway-freq-x) SWAY_FREQ_X="$2"; shift 2 ;;
    --sway-freq-y) SWAY_FREQ_Y="$2"; shift 2 ;;
    --onset-delta) ONSET_DELTA="$2"; shift 2 ;;
    --boom-amplitude) BOOM_AMPLITUDE="$2"; shift 2 ;;
    --boom-attack) BOOM_ATTACK="$2"; shift 2 ;;
    --boom-decay) BOOM_DECAY="$2"; shift 2 ;;
    --boom-min-strength) BOOM_MIN_STRENGTH="$2"; shift 2 ;;
    --no-boom) NO_BOOM_FLAG="--no-boom"; shift ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done
[[ -z "$SONG" || -z "$COVER" || -z "$OUT" ]] && { echo "need --song --cover --out"; exit 1; }

SCRATCH="$(mktemp -d)"
SLUG=$(echo "$OUT" | tr ' ' '_' | tr -cd '[:alnum:]_')
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SONG")
echo "=== $OUT ($DURATION s) ==="

echo "STAGE: extract_audio"
ffmpeg -y -i "$SONG" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$SCRATCH/audio.wav" -loglevel error

echo "STAGE: transcribe"
if [[ "$TRANSCRIBE_LOCAL" == "1" ]]; then
  python3 "$TOOLS/transcribe.py" "$SCRATCH/audio.wav" --out "$SCRATCH/words.json"
  python3 "$TOOLS/beat_detect.py" "$SCRATCH/audio.wav" --out "$SCRATCH/beats.json" --delta "$ONSET_DELTA"
else
  scp "$SCRATCH/audio.wav" "$WHISPER_HOST:/tmp/${SLUG}_audio.wav"
  ssh "$WHISPER_HOST" "LD_LIBRARY_PATH=\$(echo ~/ai-companion/comfyui-env/lib/python3.12/site-packages/nvidia/{cublas,cudnn}/lib | tr ' ' ':') ~/ai-companion/whisper-env/bin/python /tmp/lyric-video-tools/transcribe.py /tmp/${SLUG}_audio.wav --out /tmp/${SLUG}_words.json"
  ssh "$WHISPER_HOST" "~/ai-companion/whisper-env/bin/python /tmp/lyric-video-tools/beat_detect.py /tmp/${SLUG}_audio.wav --out /tmp/${SLUG}_beats.json --delta $ONSET_DELTA"
  scp "$WHISPER_HOST:/tmp/${SLUG}_words.json" "$SCRATCH/words.json"
  scp "$WHISPER_HOST:/tmp/${SLUG}_beats.json" "$SCRATCH/beats.json"
fi

# Arrays, not strings -- optional flags whose *values* contain spaces (a
# --lyrics path under a space-containing directory, for instance) get
# word-split and corrupted if these are interpolated as bare strings instead.
echo "STAGE: karaoke"
KARAOKE_EXTRA=()
[[ -n "$LYRICS" ]] && KARAOKE_EXTRA+=(--lyrics "$LYRICS")
[[ -n "$CAPS_FLAG" ]] && KARAOKE_EXTRA+=("$CAPS_FLAG")
[[ -n "$LINE_KARAOKE_FLAG" ]] && KARAOKE_EXTRA+=("$LINE_KARAOKE_FLAG")
python3 "$TOOLS/karaoke_ass.py" "$SCRATCH/words.json" "$DURATION" "$SCRATCH/lyrics.ass" \
  "${KARAOKE_EXTRA[@]}" --font "$FONT" --fontsize "$FONTSIZE" \
  --highlight-color "$HIGHLIGHT" --base-color "$BASE" \
  --align "$ALIGN" --margin-v "$MARGIN_V" --buffer "$BUFFER" --min-display "$MIN_DISPLAY"

echo "STAGE: scene"
SCENE_EXTRA=()
[[ -n "$SMOKE_FLAG" ]] && SCENE_EXTRA+=("$SMOKE_FLAG")
[[ -n "$NO_SMOKE_FLAG" ]] && SCENE_EXTRA+=("$NO_SMOKE_FLAG")
[[ -n "$NO_PULSE_FLAG" ]] && SCENE_EXTRA+=("$NO_PULSE_FLAG")
[[ -n "$NO_BOOM_FLAG" ]] && SCENE_EXTRA+=("$NO_BOOM_FLAG")
python3 "$TOOLS/scene_compositor.py" "$COVER" "$SCRATCH/beats.json" "$DURATION" "$SCRATCH/frames" \
  "${SCENE_EXTRA[@]}" \
  --zoom-drift "$ZOOM_SPEED" --sway-x "$SWAY_X" --sway-y "$SWAY_Y" \
  --sway-freq-x "$SWAY_FREQ_X" --sway-freq-y "$SWAY_FREQ_Y" \
  --boom-amplitude "$BOOM_AMPLITUDE" --boom-attack "$BOOM_ATTACK" --boom-decay "$BOOM_DECAY" \
  --boom-min-strength "$BOOM_MIN_STRENGTH" \
  --burst-count-max "$BURST_COUNT_MAX" --burst-size-max "$BURST_SIZE_MAX" --burst-speed "$BURST_SPEED" \
  --burst-area-x0 "$BURST_AREA_X0" --burst-area-x1 "$BURST_AREA_X1" \
  --burst-area-y0 "$BURST_AREA_Y0" --burst-area-y1 "$BURST_AREA_Y1" \
  --color "$PARTICLE_COLOR" \
  --jobs "$JOBS"
echo "STAGE: encode"
ffmpeg -y -framerate 24 -i "$SCRATCH/frames/frame_%06d.png" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$SCRATCH/visual.mp4" -loglevel error

FINAL="${OUTPUT_PATH:-$OUT - Lyric Video.mp4}"
echo "STAGE: mux"
ffmpeg -y \
  -i "$SCRATCH/visual.mp4" \
  -i "$SONG" \
  -filter_complex "[0:v]ass=$SCRATCH/lyrics.ass[outv]" \
  -map "[outv]" -map 1:a \
  -t "$DURATION" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a aac -b:a 192k \
  "$FINAL" -loglevel error

rm -rf "$SCRATCH"
echo "STAGE: done"
echo "done: $FINAL"
