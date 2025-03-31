1. In pipeline.sh, add the new langs to the TGT_LANGS variable
2. **TODO** Fix so that all the 3 translations produce a new index.csv to be merged with the previous one
3. Set line 592 in aggregate_translations.py to save to dataset_2.json
4. Run copy_split_format.py to apply the same split derived from dataset.json to the new dataset_2.json
