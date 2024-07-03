# %%

import json
import re
import sys
sys.path.append('EasyEdit')
import EasyEdit.easyeditor
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import ROMEHyperParams
import os
from utils import read_data
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # SELECT YOUR GPU HERE

# %%
subjects, en_subjects, prompts, ground_truth, targets = read_data('datasets/v3/post_proc/it.json', 'src')
hparams=ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama-7b.yaml')
hparams.device = 0
editor=BaseEditor.from_hparams(hparams)
max_edits = 100
metrics, edited_model, _ = editor.edit(
    prompts=prompts[:max_edits],
    ground_truth=ground_truth[:max_edits],
    target_new=targets[:max_edits],
    subject=subjects[:max_edits],
    keep_original_weight=False
)
print(metrics)
# %%
