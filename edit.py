# %%

import json
import re
import sys
sys.path.append('EasyEdit')
import EasyEdit.easyeditor
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # SELECT YOUR GPU HERE

def read_data(json_path, lang):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    subjects = []
    prompts = []
    ground_truth = []
    edits = []
    
    for key, value in data.items():
        subj_count = 0
        for relation_type, relations in value['relations'].items():
            for relation in relations:
                prompt_key = f'prompt_{lang}' if lang != "en" else "prompt"
                
                if 'edit' in relation:
                    if prompt_key in relation['edit']:
                        prompts.append(relation['edit'][prompt_key])
                        
                    ground_truth_key = f'target_sense_{lang}'
                    edit_key = f'target_sense_{lang}'
                    
                    if ground_truth_key in relation:
                        ground_truth.append(relation[ground_truth_key])
                    if edit_key in relation['edit']:
                        edits.append(relation['edit'][edit_key])
                        subj_count += 1
        subjects.extend([value['subject_senses'][f'sense_{lang}']]*subj_count)                        
    return subjects, prompts, ground_truth, edits

def clean(sense):
    # Replace underscores with spaces
    sense = sense.replace('_', ' ')
    
    # Remove round brackets and everything in between
    sense = re.sub(r'\(.*?\)', '', sense)
    
    # Remove double quotes if they wrap the entire string
    if sense.startswith('"') and sense.endswith('"'):
        sense = sense[1:-1]
    
    return sense

if __name__ == "__main__":
    subjects, prompts, ground_truth, targets = read_data('datasets/v2/it.json', 'en')
    subjects = [clean(x) for x in subjects]
    ground_truth = [clean(x) for x in ground_truth]
    targets = [clean(x) for x in targets]
    # %%
    hparams=ROMEHyperParams.from_hparams('hparams/ROME/bloom.yaml')
    hparams.device = 0
    editor=BaseEditor.from_hparams(hparams)
    max_edits = 10

    print(len(prompts), len(targets))

    metrics, edited_model, _ = editor.edit(
        prompts=prompts[:max_edits],
        ground_truth=ground_truth[:max_edits],
        target_new=targets[:max_edits],
        subject=subjects[:max_edits],
        keep_original_weight=False
    )
    print(metrics)
    # %%
