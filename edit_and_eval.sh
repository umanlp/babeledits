#!/bin/bash

# Parse command line arguments
debug=false
if [[ "$1" == "--debug" ]]; then
    debug=true
fi

# Function to join array elements with commas
join_by_comma() {
    local IFS=","
    echo "$*"
}

# Read JSON file and process each language
for lang in $(jq -r 'keys[]' sample_idxs_clp/lang_to_indexes_filtered_llama_ft-m.json); do
    # Get array of indices for current language
    indices=($(jq -r ".[\"$lang\"][]" sample_idxs_clp/lang_to_indexes_filtered_llama_ft-m.json))
    
    # Calculate hours based on number of indices
    hours=${#indices[@]}
    
    # Join indices with commas
    indices_str=$(join_by_comma "${indices[@]}")

    # Debug output
    if [[ "$debug" == true ]]; then
        echo "Processing language: $lang"
        echo "Number of hours: $hours"
        echo "Indices: $indices_str"
        echo "---"
    fi

    # Create temporary job script
    cat << EOF > tmp_job_${lang}.sh
#!/bin/bash
#SBATCH --partition=gpu_single
#SBATCH --gres=gpu:1
#SBATCH --time=${hours}:00:00
#SBATCH --job-name=edit_eval_${lang}

# Run edit.py
uv run python edit.py --config-path logs/v8_rev7/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/${lang}/prompts_mt_marked --config-name config.yaml log_subdir=collapse_test/${lang}_ft-m_top5 +sample_idx=[${indices_str}] pre_edit=null +sequential=false +return_edited_weights=true generality=false locality=false portability=false

# Run exec_eval.py
uv run python exec_eval.py --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype='bfloat16' --tasks belebele_arb_Arab,belebele_deu_Latn,belebele_eng_Latn,belebele_fra_Latn,belebele_hrv_Latn,belebele_ita_Latn,belebele_jpn_Jpan,belebele_kat_Geor,belebele_mya_Mymr,belebele_zho_Hans,xquad_ar,xquad_en --batch_size auto --device cuda:0 --output_path results_${lang}_ft-m_top-5 --load_weights /ceph/tgreen/projects/babeledits/logs/collapse_test/${lang}_ft-m_top5/meta-llama_Meta-Llama-3.1-8B-Instruct/FT-M/${lang}/prompts_mt_marked/edited_weights.pkl.gz
EOF

    # Debug output for job script
    if [[ "$debug" == true ]]; then
        echo "Generated Slurm job script for $lang:"
        cat tmp_job_${lang}.sh
        echo "===========================================\n"
    else
        # Submit job only if not in debug mode
        sbatch tmp_job_${lang}.sh
    fi

    # Clean up temp file
    rm tmp_job_${lang}.sh
done
