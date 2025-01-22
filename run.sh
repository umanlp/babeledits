#!/bin/bash

# Parameters
#SBATCH --account=westai0028
#SBATCH --error=/p/project1/westai0028/babeledits/slurm_logs/%j/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=edit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tommaso.green@uni-mannheim.de
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/p/project1/westai0028/babeledits/slurm_logs/%j/%j_0_log.out
#SBATCH --partition=dc-hwai
#SBATCH --signal=USR2@120
#SBATCH --time=24:00:00

# command
srun uv run python edit.py --config-path logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/R-ROME/ka/prompts_mt_marked --config-name config.yaml log_subdir=norm_diff generality=false locality=false portability=false eval_lm=false tgt_langs=[en] eval_prompt_type=[prompts_mt_marked] +sample_idx=null +sequential=false +return_edited_weights=false pre_edit=null