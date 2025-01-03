
#!/bin/bash

# Default values
METHOD=""
LANG=""
DEVICE=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift ;;
        --lang) LANG="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if all required arguments are provided
if [ -z "$METHOD" ] || [ -z "$LANG" ] || [ -z "$DEVICE" ]; then
    echo "Usage: $0 --method METHOD --lang LANG --device DEVICE"
    exit 1
fi
# Validate model argument
if [ "$MODEL" != "llama-3-1" ] && [ "$MODEL" != "gemma2" ]; then
    echo "Invalid model specified. Allowed values are llama-3-1 or gemma2."
    exit 1
fi

# Set MODEL environment variable based on the model argument
if [ "$MODEL" == "llama-3-1" ]; then
    MODEL="meta-llama_Meta-Llama-3.1-8B-Instruct"
elif [ "$MODEL" == "gemma2" ]; then
    MODEL="google_gemma-2-9b-it"
fi

CUDA_VISIBLE_DEVICES=$DEVICE uv run python edit.py --config-path logs/v8_rev7/$MODEL/$METHOD/$LANG/prompts_mt_marked eval_lm=false tgt_langs=[$LANG] eval_prompt_type=[prompts_mt_marked] eval_lm=null +sample_idx=null pre_edit=null +sequential=false +return_edited_weights=true generality=false locality=false portability=false log_subdir=v8_rev7_WR max_edits=250

