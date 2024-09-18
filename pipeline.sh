#!/bin/bash



export VERSION="v7"
export TOPK=15000
export LANGS="af ar az bg bn de el en es et eu fa fi fr gu he hi ht hr hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh sv sr ca cs no hy da be sk uz"
export WIKI_PATH="wikipedia_data"
export NUM_SAMPLES=30000
# export DATASET_PATH="datasets/$VERSION/hard/$LANG"
# export SYNSET_PATH="synsets/$VERSION/hard/$LANG"
# export GOLD_REL_PATH="datasets/$VERSION/agg_relations_with_prompts_filtered.tsv"
# export GLOSSARY_PATH="glossaries/$VERSION/hard/$LANG"
# export GLOSSARY_ID="${LANG}_hard_${VERSION}_1"
# export TRANSLATION_REM_PATH="translations/$VERSION/hard/$LANG"
# export BABELNET_CONF="/ceph/tgreen/projects/babeledits/babelnet_conf.yml"
# export SAMPLE_SIZE=50000

# Ensure the log directory exists
mkdir -p logs/

# Log file path will be defined after parsing arguments
mkdir -p logs/${VERSION}
LOG_FILE="logs/${VERSION}/babeledits.log"
touch "$LOG_FILE"

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

while [[ $# -gt 0 ]]; do
    case "$1" in
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



log_message "Babelnet Conf = $BABELNET_CONF"
log_message "Languages = $LANGS"
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
    local error_file="logs/${VERSION}/${script_name}_error.log"
    
    # Log the command being run
    log_message "Running ${cmd[*]}\n"

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

activate_and_run_script babelnet "get_pages.py"  --top_k $TOPK --save_path $WIKI_PATH/$VERSION --langs $LANGS
activate_and_run_script babelnet "merge_datasets.py" --num_samples $NUM_SAMPLES --save_path $WIKI_PATH/$VERSION --wiki_path $WIKI_PATH/$VERSION/processed 
activate_and_run_script babelnet "get_synsets.py" --save_dir synsets/$VERSION --data_path $WIKI_PATH/$VERSION/all_langs.csv --langs $LANGS