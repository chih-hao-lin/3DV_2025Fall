#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

mkdir -p demo_data/input_images

sequences=(
  "seq007_garden_001"
  "seq008_running_001"
)

for seq in "${sequences[@]}"; do
  echo "Processing $seq ..."

  src_dir="demo_data/sloper4d/${seq}/rgb_data/${seq}_imgs"
  dst_dir="demo_data/input_images/${seq}_imgs/cam01"
  mkdir -p "$dst_dir"
  
  # Create soft links instead of copying files
  if [ -d "$src_dir" ] && [ "$(ls -A "$src_dir" 2>/dev/null)" ]; then
    for file in "$src_dir"/*; do
      if [ -f "$file" ]; then
        ln -sf "$(realpath "$file")" "$dst_dir/"
      fi
    done
    echo "Linked $seq successfully to $dst_dir"
  fi

done

echo "All sequences linked successfully!"
