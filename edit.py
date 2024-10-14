import json
import os
import sys

import yaml

from easy_edit_adaptations.logging import redirect_edit_logs

sys.path.append("EasyEdit")
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer

from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from EasyEdit.easyeditor import BaseEditor, CounterFactDataset
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from EasyEdit.easyeditor.editors.utils import get_all_acc_keys
from utils import extract
from hydra.utils import to_absolute_path
from transformers import GenerationConfig


def get_summary_metrics(all_metrics, eval_metrics):
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
        for metric_type in eval_metrics:
            mean_metrics[eval][metric_type] = dict()
            for key in ["rewrite_acc", "rewrite_ppl"]:
                if key in all_metrics[0][eval][metric_type].keys():
                    mean_metrics[eval][metric_type][key] = np.mean(
                        [metric[eval][metric_type][key] for metric in all_metrics]
                    )
            for key in ["rephrase_acc", "locality", "portability"]:
                if (
                    key in all_metrics[0][eval][metric_type].keys()
                    and all_metrics[0][eval][metric_type][key] != {}
                ):
                    mean_metrics[eval][metric_type][key] = dict()
                    for lkey in get_all_acc_keys(all_metrics):
                        metrics = [
                            metric[eval][metric_type][key][lkey]
                            for metric in all_metrics
                            if lkey in metric[eval][metric_type][key].keys()
                        ]
                        if len(metrics) > 0:
                            mean_metrics[eval][metric_type][key][lkey] = np.mean(
                                metrics
                            )
    return mean_metrics


def prompt_to_target(prompt_type, metric_type):
    if metric_type in ["reliability", "generality"]:
        if "gloss" in prompt_type:
            return "targets"
        elif "mt_marked" in prompt_type:
            return "targets_mt_marked"
        elif "mt" in prompt_type:
            return "targets_mt"
        else:
            if prompt_type in ["prompts", "prompts_gen"]:
                return "targets_mt"
            else:
                raise ValueError(f"Unknown prompt type {prompt_type}")
    if metric_type == "locality":
        if "gloss" in prompt_type:
            return "ground_truths_loc"
        elif "mt_marked" in prompt_type:
            return "ground_truths_loc_mt_marked"
        elif "mt" in prompt_type:
            return "ground_truths_loc_mt"
        else:
            if prompt_type == "prompts_loc":
                return "ground_truths_loc_mt"
            else:
                raise ValueError(f"Unknown prompt type {prompt_type}")
    if metric_type == "portability":
        if "gloss" in prompt_type:
            return "ground_truths_port"
        elif "mt_marked" in prompt_type:
            return "ground_truths_port_mt_marked"
        elif "mt" in prompt_type:
            return "ground_truths_port_mt"
        else:
            if prompt_type == "prompts_port":
                return "ground_truths_port_mt"
            else:
                raise ValueError(f"Unknown prompt type {prompt_type}")


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Edit Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    hparams = DictConfig({**cfg.model, **cfg.method})
    method = hparams.alg_name
    model_name = hparams.model_name.replace("/", "_")

    print(
        f"Running {method} on {model_name} with {cfg.dataset} on device {cfg.device}\nUsing hparams: {hparams}\nMax edits: {cfg.max_edits}"
    )
    print("Loading data")
    with open(to_absolute_path(cfg.dataset), "r", encoding="utf-8") as file:
        data = json.load(file)
    subjects = extract(data, cfg.edit_lang, cfg.subject_type)
    prompts = extract(data, cfg.edit_lang, cfg.prompt_type)
    targets = extract(data, cfg.edit_lang, cfg.target_type)
    # ground_truths = data["ground_truth"]

    print("Data loaded")
    hparams = get_hparm_class(method).from_dict_config(hparams)
    hparams.device = cfg.device

    editor = BaseEditor.from_hparams(hparams)

    if cfg.log_subdir:
        log_dir = to_absolute_path(
            f"logs/{cfg.log_subdir}/{model_name}/{method}/{cfg.edit_lang}/{cfg.prompt_type}"
        )
        redirect_edit_logs(log_dir)
    else:
        log_dir = "logs"

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

    if cfg.subject_in_prompt == "strict":
        for p, s in zip(prompts, subjects):
            assert s in p, f"Subject {s} is not present in prompt {p}"
    elif cfg.subject_in_prompt == "loose":
        en_subjects = extract(data, "en", "subjects")
        for p, s, s_en in zip(prompts, subjects, en_subjects):
            assert (
                s in p or s_en in p
            ), f"Neither subject {s} nor subject {s_en} are present in prompt {p} and subject_in_prompt is set to loose"
        for idx in enumerate(subjects):
            if subjects[idx] not in prompts[idx]:
                subjects[idx] = en_subjects[idx]

    max_edits = cfg.max_edits if cfg.max_edits is not None else len(prompts)

    if cfg.tgt_langs is not None and cfg.eval_prompt_type is not None:
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in cfg.eval_prompt_type
            for tgt_lang in cfg.tgt_langs
        ]
        portability_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = (
                f"xlt_{cfg.prompt_type}-{tgt_prompt_type}_{cfg.edit_lang}-{tgt_lang}"
            )
            target_key = prompt_to_target(tgt_prompt_type, "reliability")
            print(
                f"Evaluating reliability in {tgt_lang} using {tgt_prompt_type} with targets {target_key}"
            )
            portability_inputs.update(
                {
                    port_key: {
                        "prompt": extract(data, tgt_lang, tgt_prompt_type)[:max_edits],
                        "ground_truth": extract(data, tgt_lang, target_key)[:max_edits],
                    },
                }
            )
    else:
        portability_inputs = None
    if cfg.generality:
        gen_prompt_types = []
        if "prompts_mt" in cfg.eval_prompt_type:
            gen_prompt_types.append("prompts_gen_mt")
        if "prompts_mt_marked" in cfg.eval_prompt_type:
            gen_prompt_types.append("prompts_gen_mt_marked")
        if "prompts_gloss" in cfg.eval_prompt_type:
            gen_prompt_types.append("prompts_gen_gloss")
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in gen_prompt_types
            for tgt_lang in cfg.tgt_langs
        ]
        generality_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            gen_key = f"{tgt_prompt_type}_{cfg.edit_lang}-{tgt_lang}"
            target_key = prompt_to_target(tgt_prompt_type, "generality")
            print(
                f"Evaluating generality in {tgt_lang} using {tgt_prompt_type} with targets {target_key}"
            )
            generality_inputs.update(
                {
                    gen_key: {
                        "prompt": extract(data, tgt_lang, tgt_prompt_type)[:max_edits],
                        "ground_truth": extract(data, tgt_lang, target_key)[:max_edits],
                    }
                }
            )
    else:
        generality_inputs = None
    if cfg.locality:
        loc_prompt_types = []
        if "prompts_mt" in cfg.eval_prompt_type:
            loc_prompt_types.append("prompts_loc_mt")
        if "prompts_mt_marked" in cfg.eval_prompt_type:
            loc_prompt_types.append("prompts_loc_mt_marked")
        if "prompts_gloss" in cfg.eval_prompt_type:
            loc_prompt_types.append("prompts_loc_gloss")
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in loc_prompt_types
            for tgt_lang in cfg.tgt_langs
        ]
        locality_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            loc_key = f"{tgt_prompt_type}_{cfg.edit_lang}-{tgt_lang}"
            target_key = prompt_to_target(tgt_prompt_type, "locality")
            print(
                f"Evaluating locality in {tgt_lang} using {tgt_prompt_type} with targets {target_key}"
            )
            locality_inputs.update(
                {
                    loc_key: {
                        "prompt": extract(
                            data, tgt_lang, tgt_prompt_type, strict=False
                        )[:max_edits],
                        "ground_truth": extract(
                            data, tgt_lang, target_key, strict=False
                        )[:max_edits],
                    }
                }
            )
    else:
        locality_inputs = None

    if cfg.portability:
        port_prompt_types = []
        if "prompts_mt" in cfg.eval_prompt_type:
            port_prompt_types.append("prompts_port_mt")
        if "prompts_mt_marked" in cfg.eval_prompt_type:
            port_prompt_types.append("prompts_port_mt_marked")
        if "prompts_gloss" in cfg.eval_prompt_type:
            port_prompt_types.append("prompts_port_gloss")

        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in port_prompt_types
            for tgt_lang in cfg.tgt_langs
        ]

        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = f"multi-hop_{tgt_prompt_type}_{cfg.edit_lang}-{tgt_lang}"
            target_key = prompt_to_target(tgt_prompt_type, "portability")
            print(
                f"Evaluating multi-hop portability in {tgt_lang} using {tgt_prompt_type} with targets {target_key}"
            )
            portability_inputs.update(
                {
                    port_key: {
                        "prompt": extract(
                            data, tgt_lang, tgt_prompt_type, strict=False
                        )[:max_edits],
                        "ground_truth": extract(
                            data, tgt_lang, target_key, strict=False
                        )[:max_edits],
                    }
                }
            )

        subj_prompt_types = ["prompts_subj_alias"]
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in subj_prompt_types
            for tgt_lang in cfg.tgt_langs
        ]
        target_key = cfg.target_type

        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = f"subj-alias_{tgt_prompt_type}_{cfg.edit_lang}-{tgt_lang}"
            print(
                f"Evaluating subj-alias portability in {tgt_lang} using {tgt_prompt_type} with targets {target_key}"
            )
            portability_inputs.update(
                {
                    port_key: {
                        "prompt": extract(
                            data, tgt_lang, tgt_prompt_type, strict=False
                        )[:max_edits],
                        "ground_truth": extract(
                            data, tgt_lang, target_key, strict=False
                        )[:max_edits],
                    }
                }
            )

    if cfg.decoding_strategy:
        gen_cfg_dict = dict(cfg.decoding_strategy)
        gen_cfg_dict.pop("type")
        gen_cfg = GenerationConfig(**gen_cfg_dict)
        print(f"Using generation: {vars(gen_cfg)}")

    if method == "FT":
        metrics, _, _ = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=generality_inputs,
            target_new=targets[:max_edits],
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=False,
            keep_original_weight=True,
            eval_metrics=cfg.metrics,
            generation_conf=gen_cfg,
        )
    else:
        metrics, _, _ = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=generality_inputs,
            subject=subjects[:max_edits],
            target_new=targets[:max_edits],
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=False,
            keep_original_weight=True,
            eval_metrics=cfg.metrics,
            generation_conf=gen_cfg,
        )

    with open(to_absolute_path(os.path.join(log_dir, "results.json")), "w") as f:
        json.dump(metrics, f, indent=4)
    summary = get_summary_metrics(metrics, cfg.metrics)
    with open(to_absolute_path(os.path.join(log_dir, "summary.json")), "w") as f:
        json.dump(summary, f)

    # Save the command used to launch the script
    command = "python " + " ".join(sys.argv)
    with open(to_absolute_path(os.path.join(log_dir, "command.txt")), "w") as f:
        f.write(command)

    with open(to_absolute_path(os.path.join(log_dir, "config.yaml")), "w") as yaml_file:
        yaml.dump(
            yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.FullLoader),
            yaml_file,
            default_flow_style=False,
            default_style="",
        )

    print(">>> FINISHED <<<")
    print(f"Logs, metrics and configurations saved to {os.path.join('logs', log_dir)}")


if __name__ == "__main__":
    main()
