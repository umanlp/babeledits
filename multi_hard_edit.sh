#!/bin/bash
# ./multi_hard_edit.sh ar de es fr hr it ja nl zh

# Check if any languages are passed
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 lang1 lang2 lang3 ..."
    exit 1
fi

# Iterate over each language passed as an argument
for lang in "$@"; do
    echo "Processing language: $lang"
    python edit.py --data_file datasets/v5/hard/$lang/translated/dataset.json \
                   --hparam hparams/FT/llama-3-1-8b-hf.yaml \
                   --lang en \
                   --tgt_langs $lang \
                   --log_subdir v5_hard_prompts_$lang \
                   --prompt_type prompts \
                   --tgt_prompt_type prompts prompts_gloss \
                   --device 6 --max_edits 2
done
