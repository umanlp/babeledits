import collections
import os
from re import sub
from sys import intern
from typing import Dict, List, Optional
import numpy as np
from pyreft import ReftModel
import torch
from .reft_hparams import ReFTHyperParams
# from ..babelreft.babelreft_main import SubloreftIntervention
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
import shutil
import logging

def apply_reft_to_model(
    model: 'CustomReftModel',
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

    if model.pos_type == "all_tokens":
        intv_threshold = 0
    elif "last" in model.pos_type:
        intv_threshold = int(model.pos_type.split("_")[-1])
    else:
        raise ValueError(f"Invalid pos_type: {model.pos_type}")
    intervention_locations = torch.tensor([[list(range(0, (target_ids[0] == -100).long().cumsum(0).argmax().item()+1))[-intv_threshold:]]]) # This is the key change @felix
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


    weights_copy["reft_interventions"] = {}
    for k, v in model.interventions.items():
        intervention = v
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

@dataclass
class SimpleOutput:
    logits: Any

class CustomReftModel(ReftModel):
    def __init__(
        self,
        reft_config,
        model,
        tokenizer,
        edited_facts_for_debug: List[str] | None = None,
        pos_type: str = "all_tokens",
        *args,
        **kwargs,
    ):
        super(CustomReftModel, self).__init__(reft_config, model)
        self.tokenizer = tokenizer
        self.edited_facts_for_debug = edited_facts_for_debug
        self.pos_type = pos_type


    def forward(
        self,
        *args,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        subspaces: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        output_original_output: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **base,
    ):
        # This assume that if positional arguments are used, then the first one
        # can only be input_ids and the second one (if there is one) can only be
        # attention_mask. Thus, in cunjunction to `**base`, this means that `a`
        # will be understood as input_ids in all of the following examples, only
        # if `a` is not a dict:
        # - in forward(a, b)
        # - in forward(a, unit_locations=c)
        # - in forward(unit_locations=c, input_ids=a)
        if len(args) > 0 and isinstance(args[0], dict):
            base = {**base, **args[0]}
        else:
            if len(args) > 0:
                base["input_ids"] = args[0]
            if len(args) > 1:
                base["attention_mask"] = args[1]
        if self.model.training:
            _, intv_output = super().forward(
                base,
                sources,
                unit_locations,
                source_representations,
                subspaces,
                labels,
                output_original_output,
                return_dict,
                use_cache,
            )
            return None, intv_output

        else:
            if not unit_locations:
                if self.pos_type == "all_tokens":
                    intv_threshold = 0
                elif "last" in self.pos_type:
                    intv_threshold = int(self.pos_type.split("_")[-1])
                else:
                    raise ValueError(f"Invalid pos_type: {self.pos_type}")
                # print(f"+++ Using {intv_threshold} tokens for intervention +++")
                unit_locations = {"sources->base": (None, [[list(range(base['input_ids'].shape[-1]))[-intv_threshold:]] for _ in range(base['input_ids'].shape[0])])}
            _, intv_output = super().forward(
                base,
                sources,
                unit_locations,
                source_representations,
                subspaces,
                labels,
                output_original_output,
                return_dict,
                use_cache,
            )
            return SimpleOutput(logits=intv_output.logits)


    def generate(
        self,
        base = None,
        sources: List | None = None,
        unit_locations: Dict | None = None,
        source_representations: Dict | None = None,
        intervene_on_prompt: bool = True,
        subspaces: List | None = None,
        **kwargs,
    ):
        if base is None:
            input_ids = kwargs["input_ids"]
            del kwargs["input_ids"]
        elif isinstance(base, dict):
            input_ids = base["input_ids"]
            for k, v in base.items():
                if k != "input_ids":
                    kwargs[k] = v
        else:
            input_ids = base


        if input_ids.shape[0] > 1:
            raise NotImplementedError(f"CustomReftModel can only handle batch size of 1 for generate")

        if not unit_locations:
            if self.pos_type == "all_tokens":
                intv_threshold = 0
            elif "last" in self.pos_type:
                intv_threshold = int(self.pos_type.split("_")[-1])
            else:
                raise ValueError(f"Invalid pos_type: {self.pos_type}")
            unit_locations = {
                "sources->base": (
                    None,
                    [
                        [list(range(input_ids.shape[-1]))[-intv_threshold:]]
                        for _ in range(input_ids.shape[0])
                    ],
                )
            }

        original, output = super().generate(
            {"input_ids": input_ids},
            sources,
            unit_locations,
            source_representations,
            intervene_on_prompt,
            subspaces,
            output_original_output=self.edited_facts_for_debug is not None,
            **kwargs,
        )
        if self.edited_facts_for_debug and not torch.equal(original, output):
            original_text = self.tokenizer.decode(original[0][input_ids.shape[1]:], skip_special_tokens=True)
            output_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            logging.info(f"Generation output:\n  before intervention:\n{original_text}\n\n  after intervention:\n{output_text}")

        return output

def get_reft_model(
    model,
    reft_config,
    tokenizer,
    pos_type,
    set_device=True,
    disable_model_grads=True,
    edited_facts_for_debug=None,
):
    """
    Create an instance of BabelReFT model.
    """
    # @felix adapt, needs to be a CustomReftModel which subclasses pyreft.ReftModel, with a generate method which sets intervention_locations to last token(s)
    reft_model = CustomReftModel(reft_config, model, tokenizer, edited_facts_for_debug=edited_facts_for_debug, pos_type=pos_type)
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
        print("Using LoReft with low rank dimension", hparams.low_rank_dim)
    elif hparams.intervention_type == "noreft":
        intervention = NoreftIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=hparams.low_rank_dim,
            add_bias=False,
        )
        print("Using NoReFt with low rank dimension", hparams.low_rank_dim)
    reft_cfg = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer,  # only single layer intervention supported
                "component": "block_output",
                "unit": "pos",
                "low_rank_dimension": hparams.low_rank_dim,
                "intervention": intervention,
            }
            for layer in hparams.layers
        ]
    )

    return reft_cfg

