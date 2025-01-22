#!/bin/bash

# Set SLURM parameters based on hostname
if hostname | grep -q "jureca"; then
    SBATCH_PARAMS=(
        "--account=westai0028"
        "--error=slurm_logs/%j/%j_0_log.err"
        "--gres=gpu:1"
        "--job-name=eval_dst"
        "--mail-type=ALL"
        "--mail-user=tommaso.green@uni-mannheim.de"
        "--mem=100GB"
        "--nodes=1"
        "--ntasks-per-node=1"
        "--open-mode=append"
        "--output=slurm_logs/%j/%j_0_log.out"
        "--partition=dc-hwai"
        "--signal=USR2@120"
        "--time=5:00:00"
    )
else
    SBATCH_PARAMS=(
        "--error=slurm_logs/%j/%j_0_log.err"
        "--output=slurm_logs/%j/%j_0_log.out"
        "--gres=gpu:1"
        "--job-name=eval_dst"
        "--mail-type=ALL"
        "--mail-user=tommaso.green@uni-mannheim.de"
        "--mem=100GB"
        "--nodes=1"
        "--ntasks-per-node=1"
        "--open-mode=append"
        "--partition=gpu-single"
        "--signal=USR2@120"
        "--time=5:00:00"
    )
fi
