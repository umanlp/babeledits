import gzip
import json
import os
import sys

import torch
import yaml

from easy_edit_adaptations.logging import redirect_edit_logs

sys.path.append("EasyEdit")
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from EasyEdit.easyeditor import BaseEditor, CounterFactDataset
from EasyEdit.easyeditor.editors.utils import summary_metrics
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from utils import extract, extract_aliases


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
    if cfg.tgt_langs:
        tgt_langs = sorted(
            [x for x in cfg.tgt_langs if x != cfg.edit_lang]
        )  # TODO fix with multisource, since edit_lang will be a list
        all_langs = sorted(list(set(tgt_langs + [cfg.edit_lang])))
    else:
        tgt_langs = None
        all_langs = [cfg.edit_lang]

    # ground_truths = data["ground_truth"]
    print("Data loaded")
    hparams = get_hparm_class(method).from_dict_config(hparams)
    hparams.device = cfg.device

    editor = BaseEditor.from_hparams(hparams)

    if cfg.log_subdir:
        if method == "FT":
            method_dir_name = "FT-M" if cfg.method.objective_optimization == "target_new" else "FT-L"
        else:
            method_dir_name = method
        log_dir = to_absolute_path(
            f"logs/{cfg.log_subdir}/{model_name}/{method_dir_name}/{cfg.edit_lang}/{cfg.prompt_type}"
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
        subjects_en = extract(data, "en", "subjects")
        subjects_gls = extract(data, cfg.edit_lang, "subjects")
        for p, s, s_en, s_gls in zip(prompts, subjects, subjects_en, subjects_gls):
            assert (
                s in p or s_en in p or s_gls in p
            ), f"Neither subject {s} nor subject {s_en} are present in prompt {p} and subject_in_prompt is set to loose"
        for idx in range(len(subjects)):
            if subjects[idx] not in prompts[idx]:
                if subjects_en[idx] in prompts[idx]:
                    subjects[idx] = subjects_en[idx]
                else:
                    subjects[idx] = subjects_gls[idx]

    max_edits = cfg.max_edits if cfg.max_edits is not None else len(prompts)

    if tgt_langs is not None and cfg.eval_prompt_type is not None:
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in cfg.eval_prompt_type
            for tgt_lang in all_langs
        ]
        portability_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = (
                f"xlt-{tgt_prompt_type}-{tgt_lang}"
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
            for tgt_lang in all_langs
        ]
        generality_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            gen_key = f"{tgt_prompt_type}-{tgt_lang}"
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
            for tgt_lang in all_langs
        ]
        locality_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            loc_key = f"{tgt_prompt_type}-{tgt_lang}"
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
            for tgt_lang in all_langs
        ]

        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = f"multi-hop_{tgt_prompt_type}-{tgt_lang}"
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
            for tgt_lang in all_langs
        ]
        target_key = cfg.target_type

        for tgt_prompt_type, tgt_lang in xlt_confs:
            port_key = f"subj-alias_{tgt_prompt_type}-{tgt_lang}"
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
            portability_inputs[port_key]["prompt"] = [
                x[0] if isinstance(x, list) else x
                for x in portability_inputs[port_key]["prompt"]
            ]  # TODO fix this in the dataset

    if cfg.decoding_strategy:
        gen_cfg_dict = dict(cfg.decoding_strategy)
        gen_cfg_dict.pop("type")
        gen_cfg = GenerationConfig(**gen_cfg_dict)
        print(f"Using generation: {vars(gen_cfg)}")

    if cfg.eval_lm:
        with open(to_absolute_path(cfg.data_lm), "r", encoding="utf-8") as file:
            data_lm = json.load(file)
        prompts_lm = [x["Text"][lang] for x in data_lm for lang in all_langs]
        lm_cfg = {
            "prompts": prompts_lm,
            "batch_size": cfg.batch_size_lm,
            "langs": all_langs,
            "num_sent_per_lang": len(data_lm),
            "metric": cfg.lm_metric,
        }
    else:
        lm_cfg = None

    if cfg.multi_answer_eval:
        aliases = extract_aliases(data, cfg.edit_lang, tgt_langs)
    else:
        aliases = None

    pre_file = to_absolute_path(os.path.join(log_dir, cfg.pre_file)) if cfg.pre_file is not None else None
    if cfg.pre_edit is not None:
        print(f"Loading pre-edit metrics from {cfg.pre_edit}")
        pre_file_path = Path(to_absolute_path(cfg.pre_edit))
        pre_file_lang = [x for x in all_langs if x in pre_file_path.parts][0]
        with gzip.open(pre_file_path, "rt") as f:
            pre_edit = json.load(f)
        pre_edit = pre_edit if cfg.max_edits is None else pre_edit[:cfg.max_edits]
        if pre_file_lang != cfg.edit_lang:
            print(f"Pre-edit file is in {pre_file_lang}, but edit language is {cfg.edit_lang}. Adjustment will be made.")
        for evaluation in pre_edit:
            for loc_key in evaluation['pre']['locality']:
                # creating tensors for logprobs
                evaluation['pre']['locality'][loc_key]['nkl']['logprobs'] = torch.tensor(evaluation['pre']['locality'][loc_key]['nkl']['logprobs'])
            if pre_file_lang != cfg.edit_lang: # if pre_file is in a different language, we need to change the key
                evaluation["pre"]["rewrite_acc"] = evaluation["pre"]["portability"][f"xlt-{cfg.prompt_type}-{cfg.edit_lang}"]
    else:
        pre_edit = None

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
            locality_metrics=cfg.locality_metrics,
            lm_cfg=lm_cfg,
            aliases=aliases,
            edit_lang=cfg.edit_lang,
            pre_file=pre_file, # TODO add to batch_edit
            pre_edit=pre_edit,
            pre_eval_only=cfg.pre_eval_only
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
            locality_metrics=cfg.locality_metrics,
            lm_cfg=lm_cfg,
            aliases=aliases,
            edit_lang=cfg.edit_lang,
            pre_file=pre_file,
            pre_edit=pre_edit,
            pre_eval_only=cfg.pre_eval_only
        )

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
    
    if cfg.pre_eval_only:
        print(">>> PRE-EVALUTION FINISHED <<<")
    else:
        with open(to_absolute_path(os.path.join(log_dir, "results.json")), "w") as f:
            json.dump(metrics, f, indent=4)
        if cfg.eval_lm:
            lm_metric = cfg.lm_metric 
        else:
            lm_metric = None
        summary = summary_metrics(metrics, cfg.metrics, cfg.locality_metrics, lm_metric=lm_metric)
        with open(to_absolute_path(os.path.join(log_dir, "summary.json")), "w") as f:
            json.dump(summary, f, indent=4)

    print(">>> FINISHED <<<")
    print(f"Logs, metrics and configurations saved to {os.path.join('logs', log_dir)}")


if __name__ == "__main__":
    main()
