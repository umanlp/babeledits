#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      BASE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$BASE_DIR" ]; then
  echo "Error: --path argument is required"
  echo "Usage: ./run_all_evals.sh --path /path/to/experiments"
  exit 1
fi

# Source SLURM parameters
source "$(dirname "$0")/slurm_config.sh"

# Find all directories containing config.yaml
find "$BASE_DIR" -name "config.yaml" -exec dirname {} \; | while read dir; do
    echo "Submitting evaluation for: $dir"
    sbatch "${SBATCH_PARAMS[@]}" eval.sh --path "$dir"
    # Wait 2 seconds between submissions
    sleep 2
done
