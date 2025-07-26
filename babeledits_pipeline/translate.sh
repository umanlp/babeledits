#!/bin/bash

mamba init
mamba activate babelnet
export DATASET_PATH="datasets/v6_marked/dataset.json"
export VERSION="v6_marked"
export GLOSSARY_ID="multi_v6" #"multi_"${VERSION} TODO CORRECT
export LANGS="ar de es fr hr it ja nl sw zh"
export OUTPUT_DIR="datasets"/${VERSION}/"translations"

python translate.py  --dataset_path $DATASET_PATH --src_blob_path "translations"/${VERSION} --tgt_blob_path "translations"/${VERSION} --glossary_id $GLOSSARY_ID -d --tgt_langs $LANGS --locality --rephrase
python translate_entities.py --dataset_path $DATASET_PATH  --src_blob_path "translations"/${VERSION} --tgt_blob_path "translations"/${VERSION} -d --tgt_langs $LANGS
python translate_with_markers.py --dataset_path $DATASET_PATH  --src_blob_path "translations_marked"/${VERSION} --tgt_blob_path "translations_marked"/${VERSION} -d --tgt_langs $LANGS --locality --rephrase
python aggregate_translations.py --dataset_path $DATASET_PATH --translation_path "datasets"/${VERSION}"/tsv/tgt" --entity_path "datasets"/${VERSION}"/tsv_entities/tgt" --marked_translation_path "datasets/"${VERSION}"/tsv_marked/tgt"