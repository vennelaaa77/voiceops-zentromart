#!/usr/bin/env bash
# Usage: ./run.sh <audio_folder> [output_folder]
set -e
cd "$(dirname "$0")"
python run.py --audio-dir "${1:?usage: ./run.sh <audio_folder> [output_folder]}" --out "${2:-sample_outputs}"
