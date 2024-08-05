#!/bin/bash

# Default values for options
device="cuda:0"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --langs)
            shift
            # Collect all following arguments as languages
            languages=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                languages+=("$1")
                shift
            done
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

# Check if languages are provided
if [ ${#languages[@]} -eq 0 ]; then
    echo "No languages provided. Use --langs to specify languages."
    exit 1
fi

# Define the path to your script
script="./hard_pipeline.sh"

# Loop through each language and execute the script
for lang in "${languages[@]}"; do
    echo "Running pipeline for language: $lang"
    $script --lang "$lang" --device "$device"
done
