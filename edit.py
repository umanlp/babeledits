

import os
import sys
import json
sys.path.append('EasyEdit')
from EasyEdit.easyeditor import BaseEditor
from utils import read_data
from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from easy_edit_adaptations.logging import redirect_edit_logs

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="datasets/v3/post_proc/it.json")
    parser.add_argument("--hparam", type=str, default='hparams/ROME/llama-7b.yaml')
    parser.add_argument("--max_edits", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_subdir", type=str, default=None)
    args = parser.parse_args()

    method = os.path.basename(os.path.dirname(args.hparam))

    subjects, en_subjects, prompts, ground_truth, targets = read_data(args.data_file, 'src')
    hparams= get_hparm_class(method).from_hparams(args.hparam)
    hparams.device = args.device

    editor=BaseEditor.from_hparams(hparams)

    if args.log_subdir:
        redirect_edit_logs(args.log_subdir)
    
    max_edits = args.max_edits
    metrics, edited_model, _ = editor.edit(
        prompts=prompts[:max_edits],
        ground_truth=ground_truth[:max_edits],
        target_new=targets[:max_edits],
        subject=subjects[:max_edits],
        keep_original_weight=False
    )
    
    if args.log_subdir:
        with open(os.path.join("logs", args.log_subdir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
