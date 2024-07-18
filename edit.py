import os
import sys
import json

sys.path.append("EasyEdit")
from utils import read_data
from EasyEdit.easyeditor import BaseEditor, CounterFactDataset
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from easy_edit_adaptations.logging import redirect_edit_logs
from sentence_transformers import SentenceTransformer
from pathlib import Path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default="datasets/v4/translated/all_langs.json"
    )
    parser.add_argument("--hparam", type=str, default="hparams/ROME/llama-7b.yaml")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--max_edits", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_subdir", type=str, default=None)
    args = parser.parse_args()

    method = os.path.basename(os.path.dirname(args.hparam))

    print(
        f"Running {method} on {args.data_file} on device {args.device}\nUsing hparams: {args.hparam}\nLogging to: {args.log_subdir}\nMax edits: {args.max_edits}"
    )
    print("Loading data")
    subjects, en_subjects, prompts, ground_truth, targets = read_data(
        args.data_file, args.lang
    )
    print("Data loaded")
    hparams = get_hparm_class(method).from_hparams(args.hparam)
    hparams.device = args.device

    editor = BaseEditor.from_hparams(hparams)

    if args.log_subdir:
        redirect_edit_logs(args.log_subdir)

    # Create train_ds if necessary
    if method == "IKE":
        fname = 'EasyEdit/data/counterfact/counterfact-train.json'
        if not os.path.isfile(fname):
            raise Exception(f"method {method} requires to download counterfactual dataset for EasyEdit. Please download the data directory and put it in EasyEdit. Link here:\nhttps://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing")
        train_ds = CounterFactDataset(fname)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        Path(os.path.join(hparams.results_dir, method, "embedding")).mkdir(parents=True, exist_ok=True)
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    
    max_edits = args.max_edits
    print(f"Editing {len(prompts)} examples")
    idxs_to_remove = []
    for idx, elem in enumerate(zip(prompts, subjects)):
        p, s = elem
        if s not in p:
            idxs_to_remove.append(idx)

    for idx in reversed(idxs_to_remove):
        del prompts[idx]
        del subjects[idx]
        del ground_truth[idx]
        del targets[idx]

    print(f"Editing {len(prompts)} examples")
    metrics, edited_model, _ = editor.edit(
        prompts=prompts[:max_edits],
        ground_truth=ground_truth[:max_edits],
        target_new=targets[:max_edits],
        subject=subjects[:max_edits],
        train_ds=train_ds,
        sequential_edit=False,
        keep_original_weight=True,
    )

    if args.log_subdir:
        with open(os.path.join("logs", args.log_subdir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
