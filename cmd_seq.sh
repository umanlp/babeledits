
declare -A dict=(
    [100]=5
    [250]=7
    [500]=15
    [1042]=20
)


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --method)
            METHOD="$2"
            if [[ "$METHOD" != "ft-m" && "$METHOD" != "ft-l" && "$METHOD" != "r-rome" && "$METHOD" != "babelreft"  && "$METHOD" != "grace"]]; then
                echo "Invalid method: $METHOD"
                exit 1
            fi
            shift
            shift
            ;;
        --num)
            NUM_EDITS="$2"
            if ! [[ "$NUM_EDITS" =~ ^[0-9]+$ ]]; then
                echo "Invalid number of edits: $NUM_EDITS"
                exit 1
            fi
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done


if [[ -v dict[$NUM_EDITS] ]]; then
    MIN=$(( dict[$NUM_EDITS] * 60 ))
else
    echo "Invalid NUM_EDITS key: $NUM_EDITS"
    exit 1
fi
echo "Number of edits: $NUM_EDITS"
echo "Max time: $MIN (minutes)"

export PLATFORM="helix"
if [[ "$METHOD" == "ft-m" ]]; then
    echo "Method: $METHOD"
    uv run python edit.py -m hydra/launcher=${PLATFORM}_min hydra.launcher.timeout_min=${MIN} log_subdir=v8_rev7_seq_${NUM_EDITS} max_edits=${NUM_EDITS} model=llama-3-1 method=ft edit_lang=en pre_edit=logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/ppl_test_set.json.gz sequential=true return_edited_weights_at_end=true method.objective_optimization=target_new method.layers=[21] method.lr=0.0005 
fi

if [[ "$METHOD" == "ft-l" ]]; then
    echo "Method: $METHOD"
    uv run python edit.py -m hydra/launcher=${PLATFORM}_min hydra.launcher.timeout_min=${MIN} log_subdir=v8_rev7_seq_${NUM_EDITS} max_edits=${NUM_EDITS} model=llama-3-1 method=ft edit_lang=en pre_edit=logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/ppl_test_set.json.gz sequential=true return_edited_weights_at_end=true method.objective_optimization=prompt_last method.layers=[19] method.lr=0.0001 method.norm_constraint=0.002
fi

if [[ "$METHOD" == "r-rome" ]]; then
    echo "Method: $METHOD"
    uv run python edit.py -m hydra/launcher=${PLATFORM}_min hydra.launcher.timeout_min=${MIN} log_subdir=v8_rev7_seq_${NUM_EDITS} max_edits=${NUM_EDITS} model=llama-3-1 method=r-rome subject_in_prompt=loose edit_lang=en pre_edit=logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/ppl_test_set.json.gz sequential=true return_edited_weights_at_end=true method.kl_factor=1 method.layers=[17]
fi

if [[ "$METHOD" == "babelreft" ]]; then
    echo "Method: $METHOD"
    uv run python edit.py -m hydra/launcher=${PLATFORM}_min hydra.launcher.timeout_min=${MIN} log_subdir=v8_rev7_seq_${NUM_EDITS} max_edits=${NUM_EDITS} model=llama-3-1 method=babelreft subject_in_prompt=loose edit_lang=en pre_edit=logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/ppl_test_set.json.gz sequential=true return_edited_weights_at_end=true method.layers=[12] method.low_rank_dim=64 method.lr=0.002
fi

if [[ "$METHOD" == "grace" ]]; then
    echo "Method: $METHOD"
    uv run python edit.py -m hydra/launcher=${PLATFORM}_min hydra.launcher.timeout_min=${MIN} log_subdir=v8_rev7_seq_${NUM_EDITS} max_edits=${NUM_EDITS} model=llama-3-1 method=grace edit_lang=en pre_edit=logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/en/prompts_mt_marked/ppl_test_set.json.gz sequential=true return_edited_weights_at_end=true method.edit_lr=0.1 method.eps=100 method.layers=[21] method.replacement=replace_all
fi