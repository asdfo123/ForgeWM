#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$PIPELINE_DIR/data/main_table/constant_action}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PIPELINE_DIR/data/main_table/constant_action_aligned77}"
TARGET_FRAMES="${TARGET_FRAMES:-77}"
MODELS=(forgewm mg2 hyworld_aligned)

source_video_dir() {
  local model="$1"
  if [[ "$model" == "hyworld_aligned" ]]; then
    printf '%s\n' "$SOURCE_ROOT/$model/eval_videos"
  else
    printf '%s\n' "$SOURCE_ROOT/$model/videos"
  fi
}

for model in "${MODELS[@]}"; do
  source_dir="$(source_video_dir "$model")"
  output_dir="$OUTPUT_ROOT/$model/videos"
  image_dir="$OUTPUT_ROOT/$model/images"
  mkdir -p "$output_dir" "$image_dir" "$OUTPUT_ROOT/$model/eval"

  # Hard-link the unchanged conditioning images when possible.
  cp -al "$SOURCE_ROOT/$model/images/." "$image_dir/"

  while IFS= read -r source_video; do
    output_video="$output_dir/$(basename "$source_video")"
    frames=""
    [[ -f "$output_video" ]] && frames="$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=nb_frames -of default=nw=1:nk=1 "$output_video")"
    if [[ "$frames" != "$TARGET_FRAMES" ]]; then
      temporary="$output_video.partial.mp4"
      ffmpeg -nostdin -v error -y -i "$source_video" -map 0:v:0 \
        -frames:v "$TARGET_FRAMES" -c copy -an "$temporary"
      mv "$temporary" "$output_video"
    fi
  done < <(find -L "$source_dir" -maxdepth 1 -type f -name '*.mp4' | sort)
done

for model in "${MODELS[@]}"; do
  video_dir="$OUTPUT_ROOT/$model/videos"
  count=0
  while IFS= read -r video; do
    IFS=',' read -r width height fps frames < <(ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$video")
    if [[ "$width" != "640" || "$height" != "352" || "$fps" != "12/1" || \
          "$frames" != "$TARGET_FRAMES" ]]; then
      echo "Invalid aligned video: $video ($width x $height, $fps, $frames frames)" >&2
      exit 1
    fi
    count=$((count + 1))
  done < <(find "$video_dir" -maxdepth 1 -type f -name '*.mp4' | sort)
  [[ "$count" == "462" ]] || { echo "$model has $count videos, expected 462" >&2; exit 1; }
  echo "$model: 462 videos, all 640x352, 12 FPS, $TARGET_FRAMES frames"
done

echo "Prepared aligned constant-action suite: $OUTPUT_ROOT"
