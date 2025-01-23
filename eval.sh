#!/bin/bash
#SBATCH --error=slurm_logs/%j/%j_0_log.err
#SBATCH --output=slurm_logs/%j/%j_0_log.out
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_dst
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.green@uni-mannheim.de
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpu-single
#SBATCH --signal=USR2@120
#SBATCH --time=5:00:00

EVAL_DIR=$1

# Read model name from config.yaml
MODEL_NAME=$(grep "model_name:" "$EVAL_DIR/config.yaml" | sed 's/.*model_name: //')

# Execute evaluation
uv run python exec_eval.py \
  --model hf \
  --model_args pretrained=$MODEL_NAME,dtype='bfloat16' \
  --tasks belebele_arb_Arab,belebele_deu_Latn,belebele_eng_Latn,belebele_fra_Latn,belebele_hrv_Latn,belebele_ita_Latn,belebele_jpn_Jpan,belebele_kat_Geor,belebele_mya_Mymr,belebele_zho_Hans,xquad_ar,xquad_de,xquad_en,xquad_zh \
  --batch_size auto \
  --device cuda:0 \
  --output_path $EVAL_DIR \
  --load_weights $EVAL_DIR/edited_weights.pkl.gz
