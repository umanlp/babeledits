python get_edits.py --lang en --output_folder datasets/v3 --rel_path datasets/v3/agg_relations_with_prompts_postedit.tsv --synset_path synsets/v3
python upload_glossary.py --source_file_name translation/glossaries/v3/glossary_en-de.csv --destination_blob_name en-de_v3.csv --glossary_id en-de_v3

it de fr hr es eu ja zh ar sw--data_file datasets/v4/translated/all_langs.json --max_edits 2000 --device 1 --log_subdir log_test --lang it
--data_file datasets/v4/translated/all_langs.json --max_edits 2000 --device 1 --log_subdir log_test --lang it
CIAO
time python edit.py --data_file datasets/v4/translated/all_langs.json --max_edits 100 --device 1 --log_subdir log_test_100_en --lang en

#
python edit_hard.py --data_file datasets/v4/translated/movies/fr.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang fr --log_subdir log_fr_hard_cs --device 0 
python edit_hard.py --data_file datasets/v4/translated/movies/it.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang it --log_subdir log_it_hard_cs --device 1 
python edit_hard.py --data_file datasets/v4/translated/movies/ja.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang ja --log_subdir log_ja_hard_cs --device 2 
python edit_hard.py --data_file datasets/v4/translated/movies/de.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang de --log_subdir log_de_hard_cs --device 3 
python edit.py --data_file datasets/v4/translated/movies/it.json --device 1 --log_subdir movies_en-it_gloss --lang en --tgt_lang it --prompt_type prompts --tgt_prompt_type prompts_gloss --hparam hparams/FT/llama-2-7b-hf.yaml
python edit.py --data_file datasets/v4/translated/movies/it.json --device 0 --log_subdir movies_it-it_gloss --lang it --tgt_lang it --prompt_type prompts --tgt_prompt_type prompts_gloss --hparam hparams/FT/llama-2-7b-hf.yaml

python edit.py --data_file datasets/v5/hard/it/translated/dataset.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang en --tgt_lang it --log_subdir en-it_hard_v5 --prompt_type prompts --tgt_prompt_type prompts
python edit.py --data_file datasets/v5/hard/it/translated/dataset.json --hparam hparams/FT/llama-2-7b-hf.yaml --lang en --tgt_lang it --log_subdir en-it_gl_hard_v5 --prompt_type prompts --tgt_prompt_type prompts_gloss


# hard
python get_synsets.py --save_dir synsets/v5/hard/it --data_path wikipedia_data/v5/hard/filtered/it.csv --langs en it
python get_hard_subsets.py --lang it --sample_size 10000
python filter_hard_pages.py --lang it --device cuda:0
python get_edits.py --langs en it --output_folder datasets/v5/hard/it --rel_path datasets/v5/hard/it/agg_relations_with_prompts.tsv --top_k 100 --synset_path synsets/v5/hard/it
python get_glossary.py --dataset_dir datasets/v5/hard/it --output_dir glossaries/v5/hard/it --langs en it 
python upload_glossary.py --source_file_name glossaries/v5/hard/it/glossary_no_id.csv --destination_blob_name glossaries/v5/hard/it/glossary_no_id.csv --glossary_id it_hard_v5
python translate.py --dataset_path datasets/v5/hard/it/dataset.json --src_blob_path translations/v5/hard/it --tgt_blob_path translations/v5/hard/it --glossary_id it_hard_rev_v5 --tgt_langs it --output_dir datasets/v5/hard/it/translated
python aggregate_translations.py --translation_path datasets/v5/hard/it/tsv/tgt --dataset_path datasets/v5/hard/it --output_dir datasets/v5/hard/it/translated --delete_same_prompt


#v5 creation

#12-14h for 5k, 20h for 10k
time python get_pages.py --top_k 10000 
python merge_datasets.py --num_samples 20000 --wiki_path wikipedia_data/v5/processed --save_path wikipedia_data/v5/
# 16 for 20k synsets
time python get_synsets.py --save_dir synsets/v5 --data_path wikipedia_data/v5/all_langs.csv 
time python get_relations.py --rephrase --generate_prompts --dataset_path datasets/v5 --synset_path synsets/v5 
# 1.5h for 5k, 2hourse for 20k with locality filtering
time python get_edits.py --output_folder datasets/v5 --rel_path datasets/v5/agg_relations_with_prompts_filtered.tsv --top_k 200 --synset_path synsets/v5 --rephrase --locality
python get_glossary.py --dataset_dir datasets/v5 --output_dir glossaries/v5 && python upload_glossary.py --source_file_name glossaries/v5/glossary_no_id.csv --destination_blob_name glossaries/v5/glossary_no_id.csv --glossary_id multi_v5
# 15 minutes for 10 langs
time python translate.py --dataset_path datasets/v5 --src_blob_path translations/v5 --tgt_blob_path translations/v5 --glossary_id multi_v5 --tgt_langs ar de es fr hr it ja nl sw zh --output_dir datasets/v5/translated --rephrase --locality
time python aggregate_translations.py --translation_path datasets/v5/tsv/tgt --dataset_path datasets/v5 --output_dir datasets/v5/translated

# editing v5
python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang en --tgt_langs ar de es fr hr it ja nl sw zh --log_subdir v5_prompts_en --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --rephrase --locality --device 0
python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang en --tgt_langs ar de es fr hr it ja nl sw zh --log_subdir v5_prompts_gloss_en --prompt_type prompts_gloss --tgt_prompt_type prompts prompts_gloss --rephrase --locality --device 0

# editing v5 hard

# x-lingual transfer en -> lang, with evaluation on lang with both prompts and prompts_gloss
python edit.py --data_file datasets/v5/hard/$lang/translated/dataset.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang en --tgt_langs $lang --log_subdir v5_hard_prompts_$lang --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --device 0

# monolingual lang -> lang, with evaluation on lang with both prompts and prompts_gloss
python edit.py --data_file datasets/v5/hard/$lang/translated/dataset.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang $lang --tgt_langs $lang --log_subdir v5_hard_prompts_$lang --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --device 0time python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang en --tgt_langs ar de es fr hr it ja nl sw zh --log_subdir v5_FT_prompts_en --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --rephrase --locality --device 0


# Edit v5 with FT and ROME
time python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/FT/llama-3-1-8b-hf.yaml --lang en --tgt_langs ar de es fr hr it ja nl sw zh --log_subdir v5_FT_prompts_en --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --rephrase --locality --device 0
time python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/ROME/llama-3-1-8b-hf.yaml --lang en --tgt_langs ar de es fr hr it ja nl sw zh --log_subdir v5_ROME_prompts_en --prompt_type prompts --tgt_prompt_type prompts prompts_gloss --rephrase --locality --device 5


# v6 creation
# I'm just fixing v5, so I copied synsets/v5 to v6 and wikipedia_data/v5 to v6
# Copying also relation files from datasets/v5 to datasets/v6


python edit.py --data_file datasets/v5/translated/test.json --hparam hparams/ROME/llama-3-1-8b-hf.yaml --lang en --tgt_langs it --log_subdir debug --prompt_type prompts_gloss --tgt_prompt_type prompts prompts_gloss --rephrase --locality --max_edits 2 --device 0 
python get_glossary.py --dataset_dir datasets/v6 --output_dir glossaries/v6 && python upload_glossary.py --source_file_name glossaries/v6/glossary_no_id.csv --destination_blob_name glossaries/v6/glossary_no_id.csv --glossary_id multi_v6
time python translate.py --dataset_path datasets/v6 --src_blob_path translations/v6 --tgt_blob_path translations/v6 --glossary_id multi_v6 --tgt_langs ar de es fr hr it ja nl sw zh --output_dir datasets/v6/translated --rephrase --locality
time python aggregate_translations.py --translation_path datasets/v6/tsv/tgt --dataset_path datasets/v6 --output_dir datasets/v6/translatedpython edit.py -m method=ft,rome max_edits=5 log_subdir=debugging device=1
python edit.py -m method=ft,rome max_edits=5 log_subdir=debugging device=1
python edit.py -m method=ft,rome max_edits=5 log_subdir=debugging device=1
python  edit.py -m  method=ft,rome max_edits=5 log_subdir=debugging device=1

python edit.py method=ft max_edits=2 log_subdir=debugging/2 device=0 locality=True generality=True
