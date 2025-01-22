#!/bin/bash

# Include SLURM parameters if this script is called directly with sbatch
if [[ "${0##*/}" == "${BASH_SOURCE##*/}" ]]; then
    source "$(dirname "$0")/slurm_config.sh"
    for param in "${SBATCH_PARAMS[@]}"; do
        #SBATCH $param
    done
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$PATH" ]; then
  echo "Error: --path argument is required"
  exit 1
fi

# Read model name from config.yaml using bash
MODEL_NAME=$(grep "model_name:" "$PATH/config.yaml" | sed 's/.*model_name: //')

# command
uv run python exec_eval.py \
  --model hf \
  --model_args pretrained=$MODEL_NAME,dtype='bfloat16' \
  --tasks belebele_arb_Arab,belebele_deu_Latn,belebele_eng_Latn,belebele_fra_Latn,belebele_hrv_Latn,belebele_ita_Latn,belebele_jpn_Jpan,belebele_kat_Geor,belebele_mya_Mymr,belebele_zho_Hans,xquad_ar,xquad_de,xquad_en,xquad_zh \
  --batch_size auto \
  --device cuda:0 \
  --output_path $PATH \
  --load_weights $PATH/edited_weights.pkl.gz
