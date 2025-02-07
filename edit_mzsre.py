import gzip
import json
import os
import sys
import copy
import torch
import yaml

from easy_edit_adaptations.logging import redirect_edit_logs

sys.path.append("EasyEdit")
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig

from easy_edit_adaptations.hparam_dispatch import get_hparm_class
from EasyEdit.easyeditor import BaseEditor, CounterFactDataset
from EasyEdit.easyeditor.editors.utils import summary_metrics
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from utils import extract, extract_aliases, get_babelreft_vocab
import pickle
import time


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


@hydra.main(config_path="configs", config_name="config_mzsre")
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
    if cfg.sample_idx is not None:
        sample_idx = (
            cfg.sample_idx
            if isinstance(cfg.sample_idx, ListConfig)
            else [cfg.sample_idx]
        )
        sample_keys = [list(data.keys())[idx] for idx in sample_idx]
        data = {sample_key: data[sample_key] for sample_key in sample_keys}
    subjects = extract(data, cfg.edit_lang, cfg.subject_type)
    prompts = extract(data, cfg.edit_lang, cfg.prompt_type)
    targets = extract(data, cfg.edit_lang, cfg.target_type)
    if cfg.tgt_langs:
            [x for x in cfg.tgt_langs if x != cfg.edit_lang]
    else:
        tgt_langs = None
        all_langs = [cfg.edit_lang]

    # ground_truths = data["ground_truth"]
    print("Data loaded")
    hparams = get_hparm_class(method).from_dict_config(hparams)
    hparams.device = cfg.device
    hparams.num_edits = cfg.max_edits if cfg.max_edits is not None else len(prompts)

    editor = BaseEditor.from_hparams(hparams)

    if cfg.log_subdir:
        if method == "FT":
            method_dir_name = (
                "FT-M" if cfg.method.objective_optimization == "target_new" else "FT-L"
            )
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
            assert s in p or s_en in p or s_gls in p, (
                f"Neither subject {s} nor subject {s_en} or subject {s_gls} are present in prompt {p} and subject_in_prompt is set to loose"
            )
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
            port_key = f"xlt-{tgt_prompt_type}-{tgt_lang}"
            target_key = 'targets'
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
        gen_prompt_types = ['prompts_gen']
        xlt_confs = [
            (tgt_prompt_type, tgt_lang)
            for tgt_prompt_type in gen_prompt_types
            for tgt_lang in all_langs
        ]
        generality_inputs = {}
        for tgt_prompt_type, tgt_lang in xlt_confs:
            gen_key = f"{tgt_prompt_type}-{tgt_lang}"
            target_key = targets
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

    if method == "BabelReFT":
        babelreft_vocab = get_babelreft_vocab(
            data, cfg.subject_type, cfg.edit_lang, cfg.tgt_langs
        )
    else:
        babelreft_vocab = None

    if cfg.return_edited_weights:
        print(">>> Saving edited weights")
    if cfg.return_edited_weights_at_end:
        print(">>> Saving edited weights at the last edit.")
    if method == "FT":
        metrics, _, _, edited_weights = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=generality_inputs,
            target_new=targets[:max_edits],
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=cfg.sequential,
            keep_original_weight=True,
            eval_metrics=cfg.metrics,
            locality_metrics=cfg.locality_metrics,
            lm_cfg=lm_cfg,
            edit_lang=cfg.edit_lang,
            pre_eval_only=cfg.pre_eval_only,
            babelreft_vocab=babelreft_vocab,
            return_edited_weights=cfg.return_edited_weights,
            return_edited_weights_at_end=cfg.return_edited_weights_at_end,
        )
    else:
        metrics, _, _, edited_weights = editor.edit(
            prompts=prompts[:max_edits],
            # ground_truth=ground_truth[:max_edits],
            rephrase_prompts=generality_inputs,
            subject=subjects[:max_edits],
            target_new=targets[:max_edits],
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            sequential_edit=cfg.sequential,
            keep_original_weight=True,
            eval_metrics=cfg.metrics,
            locality_metrics=cfg.locality_metrics,
            lm_cfg=lm_cfg,
            edit_lang=cfg.edit_lang,
            pre_eval_only=cfg.pre_eval_only,
            babelreft_vocab=babelreft_vocab,
            return_edited_weights=cfg.return_edited_weights,
            return_edited_weights_at_end=cfg.return_edited_weights_at_end,
        )

    print(
        f"\033[93mPeak GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB, Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GB\033[0m"
    )

    if edited_weights is not None:
        print("Saving edited weights to disk")
        with gzip.open(os.path.join(log_dir, "edited_weights.pkl.gz"), "wb") as f:
            pickle.dump(edited_weights, f)

    # Save the command used to launch the script
    command = "python " + " ".join(sys.argv)
    with open(to_absolute_path(os.path.join(log_dir, "command.txt")), "w") as f:
        f.write(command)

    with open(to_absolute_path(os.path.join(log_dir, "config.yaml")), "w") as yaml_file:
        config_dict = yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.FullLoader)
        config_dict["job_id"] = (
            os.getenv("SLURM_JOB_ID") if os.getenv("SLURM_JOB_ID") else None
        )
        yaml.dump(
            config_dict,
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

        eval_phases = (
            ["pre", "intermediate", "post"] if cfg.sequential else ["pre", "post"]
        )
        summary = summary_metrics(
            metrics,
            cfg.metrics,
            cfg.locality_metrics,
            lm_metric=lm_metric,
            eval_phases=eval_phases,
        )
        with open(to_absolute_path(os.path.join(log_dir, "summary.json")), "w") as f:
            json.dump(summary, f, indent=4)

    print(">>> FINISHED <<<")
    print(f"Logs, metrics and configurations saved to {os.path.join('logs', log_dir)}")


if __name__ == "__main__":
    main()
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Time required to execute main: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )
    job_id = os.getenv("SLURM_JOB_ID")
    print(f"SLURM_JOB_ID: {job_id}")
