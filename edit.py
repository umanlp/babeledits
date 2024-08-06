import os
import sys
import json

import yaml

sys.path.append("EasyEdit")
from utils import extract
from EasyEdit.easyeditor import BaseEditor, CounterFactDataset
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from easy_edit_adaptations.logging import redirect_edit_logs
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np


def get_summary_metrics(all_metrics):
    if isinstance(all_metrics, dict):
        all_metrics = [
            all_metrics,
        ]
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    output_file = os.path.join(logs_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    mean_metrics = dict()
    for eval in ["pre", "post"]:
        mean_metrics[eval] = dict()
        for key in ["rewrite_acc", "rephrase_acc"]:
            if key in all_metrics[0][eval].keys():
                mean_metrics[eval][key] = np.mean(
                    [metric[eval][key] for metric in all_metrics]
                )
        for key in ["locality", "portability"]:
            if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                mean_metrics[eval][key] = dict()
                for lkey in all_metrics[0][eval][key].keys():
                    if lkey.endswith("acc"):
                        mean_metrics[eval][key][lkey] = np.mean(
                            [metric[eval][key][lkey] for metric in all_metrics]
                        )
    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

    return mean_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str, default="datasets/v4/translated/all_langs.json"
    )
    parser.add_argument("--hparam", type=str, default="hparams/ROME/llama-7b.yaml")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--tgt_langs", type=str, default=None, nargs="+")
    parser.add_argument("--max_edits", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_subdir", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default=None)
    parser.add_argument("--tgt_prompt_type", type=str, default=None, nargs="+")
    parser.add_argument(
        "--rephrase", action="store_true", help="rephrase the questions"
    )
    parser.add_argument(
        "--locality", action="store_true", help="whether to also get a locality"
    )
    args = parser.parse_args()

    method = os.path.basename(os.path.dirname(args.hparam))

    print(
        f"Running {method} on {args.data_file} on device {args.device}\nUsing hparams: {args.hparam}\nLogging to: {args.log_subdir}\nMax edits: {args.max_edits}"
    )
    print("Loading data")

    with open(args.data_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    subjects = extract(data, args.lang, "subjects")
    prompts = extract(data, args.lang, "prompts")
    targets = extract(data, args.lang, "targets")
    # ground_truths = data["ground_truth"]

    print("Data loaded")
    hparams = get_hparm_class(method).from_hparams(args.hparam)
    hparams.device = args.device

    editor = BaseEditor.from_hparams(hparams)

    if args.log_subdir:
        redirect_edit_logs(args.log_subdir)

    # Create train_ds if necessary
    if method == "IKE":
        fname = "EasyEdit/data/counterfact/counterfact-train.json"
        if not os.path.isfile(fname):
            raise Exception(
                f"method {method} requires to download counterfactual dataset for EasyEdit. Please download the data directory and put it in EasyEdit. Link here:\nhttps://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing"
            )
        train_ds = CounterFactDataset(fname)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(
            f"cuda:{hparams.device}"
        )
        Path(os.path.join(hparams.results_dir, method, "embedding")).mkdir(
            parents=True, exist_ok=True
        )
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    if method == "ROME":
        print(f"Size of the data {len(prompts)}")
        idxs_to_remove = []
        for idx, elem in enumerate(zip(prompts, subjects)):
            p, s = elem
            if p.count(s) != 1:
                idxs_to_remove.append(idx)

        for idx in reversed(idxs_to_remove):
            del prompts[idx]
            del subjects[idx]
            # del ground_truth[idx]
            del targets[idx]

        print(f"Size of the dataset after filtering: {len(prompts)}")

    max_edits = args.max_edits if args.max_edits is not None else len(prompts)

    if args.tgt_langs is not None and args.tgt_prompt_type is not None:
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in args.tgt_prompt_type
            for tgt_lang in args.tgt_langs
        ]
        portability_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = f"{args.prompt_type}-{tgt_prompt_type}_{args.lang}-{tgt_lang}"
            portability_inputs.update(
                {
                    port_key: {
                        "prompt": extract(data, tgt_lang, tgt_prompt_type)[:max_edits],
                        "ground_truth": extract(data, tgt_lang, "targets")[:max_edits],
                    },
                }
            )
    else:
        portability_inputs = None
    if args.rephrase:
        rephrase_prompts = extract(data, args.lang, "prompts_gen")[:max_edits]
    else:
        rephrase_prompts = None
    if args.locality:
        locality_inputs = {}
        locality_inputs.update(
            {
                "locality": {
                    "prompt": extract(data, args.lang, "prompts_loc")[:max_edits],
                    "ground_truth": extract(data, args.lang, "ground_truths_loc")[:max_edits],
                }
            }
        )
    else:
        locality_inputs = None
    if method == "FT":
        metrics, edited_model, _ = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=rephrase_prompts,
            target_new=targets[:max_edits],
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=False,
            keep_original_weight=True,
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=rephrase_prompts,
            subject=subjects[:max_edits],
            target_new=targets[:max_edits],
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=False,
            keep_original_weight=True,
        )

    if args.log_subdir:
        with open(os.path.join("logs", args.log_subdir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        summary = get_summary_metrics(metrics)
        with open(os.path.join("logs", args.log_subdir, "summary.log"), "w") as f:
            f.write(str(summary))

        # Save the command used to launch the script
        command = "python " + " ".join(sys.argv)
        with open(os.path.join("logs", args.log_subdir, "command.txt"), "w") as f:
            f.write(command)

        hparams_dict = {
            attr: getattr(hparams, attr)
            for attr in dir(hparams)
            if not attr.startswith("__") and not callable(getattr(hparams, attr))
        }

        with open(
            os.path.join("logs", args.log_subdir, f"hparams_{method}.yaml"), "w"
        ) as yaml_file:
            yaml.dump(hparams_dict, yaml_file, default_flow_style=False)

        print(">>> FINISHED <<<")
        print(
            f"Logs, metrics and hyperparameters saved to {os.path.join('logs', args.log_subdir)}"
        )
