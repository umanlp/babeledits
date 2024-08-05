#!/bin/bash

# Ensure the log directory exists
mkdir -p logs/hard

# Log file path will be defined after parsing arguments
LOG_FILE=""

# Start the total execution timer
total_start_time=$(date +%s)

# Declare an associative array to store script times
declare -A script_times

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to send an email with error details
send_failure_email() {
    local script_name=$1
    local params=$2
    local error_file=$3
    local hostname=$(hostname)
    local failure_time=$(date "+%Y-%m-%d %H:%M:%S")

    local subject="Script Failure: $script_name"
    local body="The script '$script_name' failed on host '$hostname'.\n\nParameters: $params\nTime of failure: $failure_time\nHost: $hostname\n\nError Traceback:\n$(cat "$error_file")"

    echo -e "$body" | mail -s "$subject" "$USER_MAIL"
}

# Source the bashrc file
source ~/.bashrc

# v5 hard creation
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)
            export LANG="$2"
            shift 2
            ;;
        --device)
            export DEVICE="$2"
            shift 2
            ;;
        --user-mail)
            export USER_MAIL="$2"
            shift 2
            ;;
        *)
            log_message "Invalid argument: $1"
            exit 1
            ;;
    esac
done

# Set the log file path after LANG has been set
LOG_FILE="logs/hard/${LANG}/${LANG}.log"

export WIKI_PATH="wikipedia_data/v5/hard"
export DATASET_PATH="datasets/v5/hard/$LANG"
export SYNSET_PATH="synsets/v5/hard/$LANG"
export GOLD_REL_PATH="datasets/v5/agg_relations_with_prompts_filtered.tsv"
export GLOSSARY_PATH="glossaries/v5/hard/$LANG"
export GLOSSARY_ID="${LANG}_hard_v5_1"
export TRANSLATION_REM_PATH="translations/v5/hard/$LANG"
export BABELNET_CONF="/ceph/tgreen/projects/babeledits/babelnet_conf2.yml"

log_message "Babelnet Conf = $BABELNET_CONF"
log_message "Language = $LANG"
log_message "Device = $DEVICE"
log_message "Wiki path = $WIKI_PATH"
log_message "Dataset path = $DATASET_PATH"
log_message "Synset path = $SYNSET_PATH"
log_message "Glossary ID = $GLOSSARY_ID"
log_message "Glossary path = $GLOSSARY_PATH"
log_message "Translation REM path = $TRANSLATION_REM_PATH"
log_message "User Email = $USER_MAIL"

# Function to activate environment and run a script
activate_and_run_script() {
    local environment=$1
    local script_name=$2
    shift 2

    # Activate the conda environment
    mamba activate "$environment"

    # Capture the command and its parameters
    local cmd=("python" "$script_name" "$@")
    local start_time=$(date +%s)
    local error_file="logs/hard/${script_name}_error.log"
    
    # Log the command being run
    log_message "Running ${cmd[*]}"

    # Run the command, redirecting stderr to both the error file and stdout
    "${cmd[@]}" 2> >(tee "$error_file") || {
        log_message "Error: $script_name failed."
        send_failure_email "$script_name" "${cmd[*]}" "$error_file"
        exit 1
    }
    
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    # Use awk for floating-point division to calculate minutes
    local execution_minutes=$(echo "$execution_time / 60" | awk '{printf "%.2f", $1}')
    script_times["$script_name"]="$execution_minutes"
    log_message "$script_name took ${script_times[$script_name]} minutes"
}

# Run each script with logging and timing
activate_and_run_script babelnet "get_hard_subsets.py" --lang "$LANG" --sample_size 10000
activate_and_run_script babeledits "filter_hard_pages.py" --lang "$LANG" --device "$DEVICE" --top_k 1000
activate_and_run_script babelnet "get_synsets.py" --langs en "$LANG" --save_dir "$SYNSET_PATH" --data_path "$WIKI_PATH/filtered/$LANG.csv"
activate_and_run_script babelnet "get_relations.py" --dataset_path "$DATASET_PATH" --synset_path "$SYNSET_PATH"
activate_and_run_script babelnet "auto_filter_relations.py" --orig_rel "$DATASET_PATH/agg_relations_with_subj_obj.tsv" --gold_rel "$GOLD_REL_PATH" --output_path "$DATASET_PATH/agg_relations_with_prompts_filtered.tsv"
activate_and_run_script babelnet "get_edits.py" --langs en "$LANG" --output_folder "$DATASET_PATH" --rel_path "$DATASET_PATH/agg_relations_with_prompts_filtered.tsv" --top_k 200 --synset_path "$SYNSET_PATH"
activate_and_run_script babelnet "get_glossary.py" --langs en "$LANG" --dataset_dir "$DATASET_PATH" --output_dir "$GLOSSARY_PATH"
activate_and_run_script babelnet "upload_glossary.py" --source_file_name "$GLOSSARY_PATH/glossary_no_id.csv" --destination_blob_name "$GLOSSARY_PATH/glossary_no_id.csv" --glossary_id "$GLOSSARY_ID"
activate_and_run_script babelnet "translate.py" --dataset_path "$DATASET_PATH" --src_blob_path "$TRANSLATION_REM_PATH" --tgt_blob_path "$TRANSLATION_REM_PATH" --glossary_id "$GLOSSARY_ID" --tgt_langs "$LANG" --output_dir "$DATASET_PATH/translated"
activate_and_run_script babelnet "aggregate_translations.py" --translation_path "$DATASET_PATH/tsv/tgt" --dataset_path "$DATASET_PATH/dataset.json" --output_dir "$DATASET_PATH/translated" --delete_same_prompt

# Calculate the total execution time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time - total_start_time))
# Use bc to calculate total execution time in minutes
total_execution_minutes=$(echo "scale=2; $total_execution_time / 60" | bc)

# Log the execution times summary
log_message "Execution times (in minutes) for each script:"
for script in "${!script_times[@]}"; do
    log_message "$script: ${script_times[$script]} minutes"
done
log_message "Total execution time: $total_execution_minutes minutes"