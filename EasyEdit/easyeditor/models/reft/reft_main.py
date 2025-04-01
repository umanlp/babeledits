import collections
import os
from re import sub
from sys import intern
from typing import Dict, List, Optional
import numpy as np
from pyreft import ReftModel
import torch
from .reft_hparams import ReftHyperParams
from ..babelreft.babelreft_main import SubloreftIntervention
import pyreft
from ...util import nethook
import datasets
from pyreft.dataset import ReftDataCollator
from pyreft import ReftTrainerForCausalLM, NoreftIntervention
from pyvene import TrainableIntervention
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
    DataCollatorForSeq2Seq,
)
import copy as cp
from dataclasses import dataclass
from typing import Any
import ahocorasick
import shutil
import logging

def apply_reft_to_model(
    model: 'ReftModel',
    tok,
    requests,
    hparams,
    copy=False,
    return_orig_weights=True,
    keep_original_weight=False,
    **kwargs,
):
    weights_copy = {}

    with torch.no_grad():
        for w_name, weight in model.named_parameters():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

    request = requests[0]  # TODO should be adapted to work for multi-source
    full_input = f"{request['prompt']} {request['target_new']}"
    tok.padding_side = "right"
    input_ids = tok(request["prompt"], return_tensors="pt")["input_ids"]
    full_input_ids = tok(full_input, return_tensors="pt")["input_ids"]
    target_ids = cp.deepcopy(full_input_ids)
    target_ids[0, : input_ids.shape[-1]] = -100

    intervention_locations = torch.tensor([[[full_input_ids.shape[-1] - 1]]]) # This is the key change @felix
    data = {
        "input_ids": full_input_ids,
        "intervention_locations": intervention_locations.permute(1, 0, 2).tolist(),
        "labels": target_ids,
    }


    train_dataset = datasets.Dataset.from_dict(
        data
    )

    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    data_module = dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

    output_base = "./reft_res/"
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    existing_dirs = [d for d in os.listdir(output_base) if d.startswith("edit_")]
    if existing_dirs:
        i = max(int(d.split("_")[1]) for d in existing_dirs) + 1
    else:
        i = 0
    output_dir = os.path.join(output_base, f"edit_{i}")
    os.makedirs(output_dir)

    saving_callback = SaveBestTrainingLossCallback()

    train_module = data_module
    training_args = TrainingArguments(
        num_train_epochs=hparams.num_steps,
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=hparams.lr,
        logging_steps=1,
        report_to=[],
        save_steps=1,
        save_strategy="steps",
    )
    # breakpoint()
    trainer = ReftTrainerForCausalLM(
        model=model,
        tokenizer=tok,
        args=training_args,
        callbacks=[TrainingLossThresholdCallback(threshold=0.01), saving_callback],
        **train_module,
    )

    _ = trainer.train()

    # Loading best checkpoint
    print(f"Loading {saving_callback.best_checkpoint_path}")
    trainer.model.load_intervention(
        f"{saving_callback.best_checkpoint_path}/intervenable_model", 
        include_model=False # Not really needed since we froze the whole model, but consistent with PyREFT way
    )
    # Removing best checkpoint from disk
    shutil.rmtree(saving_callback.best_checkpoint_path)

    # breakpoint()

    weights_copy["reft_interventions"] = {}
    for k, v in model.interventions.items():
        intervention = v[0]
        if isinstance(intervention, TrainableIntervention):
            weights_copy["reft_interventions"][k] = cp.deepcopy(intervention.state_dict())
    weights_copy["reft_init"] = {
        "pos_type": model.pos_type
    }
    
    tok.padding_side = "left"
    return model, weights_copy


class TrainingLossThresholdCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is not None and "loss" in logs:
            if logs["loss"] < self.threshold:
                print(
                    f"Training loss {logs['loss']} below threshold {self.threshold}. Stopping training."
                )
                control.should_training_stop = True
        return control


class SaveBestTrainingLossCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float('inf')
        self.best_checkpoint_path = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'loss' not in logs:
            return

        current_loss = logs['loss']
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            control.should_save = True

            # Construct the path for the new best checkpoint
            step = state.global_step
            self.best_checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{step}')

            print(f"New best model saved at step {step} with loss: {self.best_loss:.4f}")
            print(f"Best checkpoint path: {self.best_checkpoint_path}")
        else:
            control.should_save = False

    def on_save(self, args, state, control, **kwargs):
        # Delete all checkpoints except the best one
        for dirname in os.listdir(args.output_dir):
            if dirname.startswith("checkpoint-"):
                checkpoint_path = os.path.join(args.output_dir, dirname)
                if checkpoint_path != self.best_checkpoint_path: # delete all prev ckpts except the best one
                    shutil.rmtree(checkpoint_path)

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training completed. Best model checkpoint: {self.best_checkpoint_path}")
        print(f"Best loss achieved: {self.best_loss:.4f}")
                


def get_reft_model(
    model,
    reft_config,
    pos_type,
    low_rank_dim,
    tokenizer,
    words_to_add=None,
    set_device=True,
    disable_model_grads=True,
    edited_facts_for_debug=None,
):
    """
    Create an instance of BabelReFT model.
    """
    # @felix adapt, needs to be a CustomReftModel which subclasses pyreft.ReftModel, with a generate method which sets intervention_locations to last token(s)
    reft_model = BabelReftModel(reft_config, model, pos_type, tokenizer, words_to_add, low_rank_dim, edited_facts_for_debug=edited_facts_for_debug)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()
    return reft_model


def get_reft_config(hparams, hidden_size):
    if hparams.intervention_type == "loreft":
        intervention = pyreft.LoreftIntervention(
            embed_dim=hidden_size, low_rank_dimension=hparams.low_rank_dim
        )
        low_rank_dim = hparams.low_rank_dim
        print("Using LoReft with low rank dimension", low_rank_dim)
    elif hparams.intervention_type == "subloreft":
        # This is for retro-compatibility
        fact_nb = hparams.num_edit_factor if hasattr(hparams, "num_edit_factor") else hparams.num_edits
        intervention = SubloreftIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=hparams.low_rank_dim * fact_nb,
            add_bias=False,
        )
        low_rank_dim = hparams.low_rank_dim * fact_nb
        print("Using SubLoReft with low rank dimension", low_rank_dim)
    reft_cfg = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer,  # only single layer intervention supported
                "component": "block_output",
                "unit": "pos",
                "low_rank_dimension": low_rank_dim,
                "intervention": intervention,
            }
            for layer in hparams.layers
        ]
    )

    return reft_cfg


if __name__ == "__main__":
    import torch, transformers, pyreft

    # from EasyEdit.easyeditor.models.babelreft.babelreft_main import get_babelreft_model
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = "cuda"

    # prompt_no_input_template = """\n<|user|>:%s</s>\n<|assistant|>:"""

    # model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device
    )
    # model_name_or_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2")

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    reft_config = pyreft.ReftConfig(
        representations={
            "layer": 4,
            "component": "block_output",
            "low_rank_dimension": 4,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size, low_rank_dimension=4
            ),
        }
    )
    babelreft_model = get_babelreft_model(
        model, reft_config, tokenizer, words_to_add=["hello", "world"]
    )
    babelreft_model.set_device("cuda")
    babelreft_model.print_trainable_parameters()

    sentences = [
        "hey hello there ! The quick brown fox jumps over the lazy dog.",
        "The most wonderful city is Beijing, which is the capital city of China.",
        "Artificial intelligence is transforming the world.",
        "The University of Padova is one of the oldest universities in the world. Also, Padova is the best.",
    ]
    babelreft_model.add_word_to_vocab("Padova")

    babelreft_model.vocab
    babelreft_model.get_vocab_words()

    tokenizer.pad_token = tokenizer.eos_token
    tok_sequences = tokenizer(sentences, padding=True, return_tensors="pt").to(device)
    babelreft_model.get_unit_locations(tok_sequences["input_ids"])

    babelreft_model(tok_sequences)

    words = babelreft_model.vocab
    print(babelreft_model.get_vocab_words())
    tok_sequences = tokenizer(sentences)["input_ids"]

    def find_word_indices(tok_sequences, words):
        result = []
        for seq in tok_sequences:
            d = []
            for idx, word_tokens in enumerate(words):
                word_len = len(word_tokens)
                for i in range(len(seq) - word_len + 1):
                    if seq[i : i + word_len] == word_tokens:
                        d.append({"word_idx": idx, "last_token_pos": i + word_len - 1})
            result.append(d or None)
        return result

    find_word_indices(tok_sequences, words)
