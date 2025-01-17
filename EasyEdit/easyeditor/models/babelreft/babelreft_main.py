import collections
import os
from re import sub
from sys import intern
from typing import Dict, List, Optional
import numpy as np
from pyreft import ReftModel
import torch
from .babelreft_hparams import BabelReFTHyperParams
import pyreft
from ...util import nethook
import datasets
from pyreft.dataset import ReftDataCollator
from pyreft import ReftTrainerForCausalLM, NoreftIntervention
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
    DataCollatorForSeq2Seq,
)
import copy as cp
import ahocorasick


def apply_babelreft_to_model(
    model,
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

    intervention_locations, _, subspaces = model.get_intervention_setup(
        full_input_ids
    )
    intervention_locations = torch.tensor(
            intervention_locations["sources->base"][1]
        )
    data = {
        "input_ids": full_input_ids,
        "intervention_locations": intervention_locations.permute(1, 0, 2).tolist(),
        "labels": target_ids,
    }
    if subspaces is not None:
        data["subspaces"] = torch.tensor(subspaces).permute(1, 0, 2).tolist()

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

    output_dir = "./reft_res/"

    saving_callback = SaveBestTrainingLossCallback(output_dir)

    train_module = data_module
    training_args = TrainingArguments(
        num_train_epochs=hparams.num_steps,
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=hparams.lr,
        logging_steps=1,
        report_to=[],
        save_steps=0.1,
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

    print(f"Loading {saving_callback.best_checkpoint}")
    trainer.model.load_intervention(
        f"{saving_callback.best_checkpoint}/intervenable_model", 
        include_model=False # Not really needed since we froze the whole model, but consistent with PyREFT way
    )

    # breakpoint()
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

    def __init__(self, output_dir: str):
        self.best_checkpoint = None
        self._best_loss = None
        self.output_dir = output_dir

    def _get_most_recent_checkpoint(self):
        elements = os.listdir(self.output_dir)
        checkpoints = [elt for elt in elements if elt.startswith("checkpoint-")]
        checkpoint_paths = [os.path.join(self.output_dir, ckpt) for ckpt in checkpoints] 
        return max(checkpoint_paths, key=os.path.getctime)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        last_loss = None
        log_history = state.log_history
        for log in log_history[::-1]:
            if "loss" in log:
                last_loss = log["loss"]
                break
        
        if last_loss is not None and (self._best_loss is None or last_loss < self._best_loss):
            print(f"New best checkpoint with loss {last_loss}")
            self._best_loss = last_loss
            self.best_checkpoint = self._get_most_recent_checkpoint()
                


def check_same_intervention_size(pos_type, unit_locations):
    """
    Check that, for each example in the batch, the number of interventions (positions on which to intervene) is the same.
    """
    if pos_type != "all_tokens":
        return True
    else:
        try:
            x = torch.tensor(unit_locations["sources->base"][1])
            return True
        except ValueError:
            return False


class BabelReftModel(ReftModel):
    def __init__(
        self,
        reft_config,
        model,
        pos_type,
        tokenizer,
        words_to_add=None,
        low_rank_dim=None,
        *args,
        **kwargs,
    ):
        super(BabelReftModel, self).__init__(reft_config, model)
        self.vocab = {}
        self.tokenizer = tokenizer
        self.pos_type = pos_type
        self.automaton = ahocorasick.Automaton()
        self.low_rank_dim = low_rank_dim
        self.word_to_subspace = collections.OrderedDict()
        if words_to_add is not None:
            self.add_words_to_vocab(words_to_add)

    def forward(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        subspaces: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        output_original_output: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ):
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
            unit_locations, intervention_mask, subspaces = self.get_intervention_setup(
                base["input_ids"]
            )
            # at least one example does not have an intervention
            og_output = self.model(**base) if not intervention_mask.all() else None
            if not intervention_mask.any():
                # Vanilla forward pass, no interventions
                return og_output.logits.bfloat16()
            elif check_same_intervention_size(self.pos_type, unit_locations):
                # all examples have the same number of interventions, so we can do a single forward pass
                _, intv_output = super().forward(
                    base=base,
                    output_original_output=False,
                    unit_locations=unit_locations,
                    subspaces=subspaces
                )
                # if only some examples have interventions, we need to replace the logits with the intervention logits
                output_logits = (
                    intv_output.logits.bfloat16()
                    if og_output is None
                    else torch.where(
                        intervention_mask[:, None, None].bool(),
                        intv_output.logits.bfloat16(),
                        og_output.logits.bfloat16(),
                    )
                )
            else:
                # examples have different number of interventions, so we need to do a forward pass for each example
                all_logits = []
                batch_size = base["input_ids"].shape[0]
                num_layers = len(self.interventions)
                for i in range(
                    batch_size
                ):  # need to do this cause pyvene does not support different number of intv across examples
                    if intervention_mask[i] == 1:
                        sel_base = {k: v[i].unsqueeze(0) for k, v in base.items()}
                        intervention_sel = slice(
                            i * num_layers,
                            (i + 1) * num_layers,
                        )
                        sel_unit_locations = {
                            "sources->base": (
                                None,
                                unit_locations["sources->base"][1][intervention_sel],
                            )
                        }
                        sel_subspace = subspaces[intervention_sel]
                        _, intv_output = super().forward(
                            base=sel_base,
                            output_original_output=False,
                            unit_locations=sel_unit_locations,
                            subspaces=sel_subspace,
                        )
                        all_logits.append(intv_output.logits.bfloat16())
                    else:
                        all_logits.append(og_output.logits[i].unsqueeze(0).bfloat16())
                output_logits = torch.cat(all_logits, dim=0)
            return output_logits

    def add_words_to_vocab(self, word_list):
        word_list = [word for word in word_list if word not in self.vocab]
        if len(word_list) == 0:
            return
        all_words = word_list + [f" {w}" for w in word_list]
        all_tokens = self.tokenizer(all_words, add_special_tokens=False)["input_ids"]
        token_seqs = all_tokens[: len(word_list)]
        token_seqs_space = all_tokens[len(word_list) :]
        self.vocab.update(
            {
                word: [seq, seq_space]
                for word, seq, seq_space in zip(word_list, token_seqs, token_seqs_space)
            }
        )
        if self.config.intervention_types[0] == SubloreftIntervention:
            if len(self.word_to_subspace) > 0:
                last_key = next(reversed(self.word_to_subspace))
                last_value = self.word_to_subspace[last_key][-1]
                self.word_to_subspace.update(
                    {
                        word: list(
                            range(
                                last_value + 1, 
                                last_value + 1 + self.low_rank_dim,
                            )
                        )
                        for word in word_list
                    }
                )
            else:
                last_key = None
                last_value = None
                self.word_to_subspace.update(
                    {
                        word: list(range(0, self.low_rank_dim))
                        for word in word_list
                    }
                )
        for idx, key in enumerate(word_list):
            self.automaton.add_word(key, (idx, key))
        self.automaton.make_automaton()

    def get_vocab_words(self):
        return list(self.vocab.keys())

    def reset_vocab(self):
        self.vocab.clear()
        if self.config.intervention_types[0] == SubloreftIntervention:
            self.word_to_subspace.clear()

    def get_intervention_setup(self, tok_sequences):
        if len(self.vocab) == 0:
            return {
                "sources->base": (None, [[[0]] for _ in range(len(tok_sequences))])
            }, torch.zeros(len(tok_sequences), device=tok_sequences.device), None
        result = []
        subspaces = []
        haystack = self.tokenizer.batch_decode(tok_sequences, skip_special_tokens=True)
        for seq, tok_seq in zip(haystack, tok_sequences):
            # print(seq, tokenizer.decode(seq))
            word_matches = []
            for _, (_, original_value) in self.automaton.iter(seq):
                word_matches.append(original_value)
                break  # only one match
            if len(word_matches) == 0:
                result.append(None)
            else:  # string match!
                token_match_found = False
                for word_tokens in self.vocab[word_matches[0]]:
                    word_len = len(word_tokens)
                    word_tokens = torch.tensor(word_tokens, device=tok_sequences.device)
                    matches = (
                        tok_seq[
                            torch.arange(len(tok_seq) - word_len + 1)[:, None]
                            + torch.arange(word_len)
                        ]
                        == word_tokens
                    ).all(axis=1)
                    if matches.any():
                        start_idx = matches.long().argmax().item()
                        if self.pos_type == "last_token":
                            pos = [start_idx + word_len - 1]
                        elif self.pos_type == "all_tokens":
                            pos = list(range(start_idx, start_idx + word_len))
                        else:
                            raise ValueError(f"Invalid pos_type: {self.pos_type}")
                        result.append(
                            [
                                {
                                    "word": word_matches[0],
                                    f"{self.pos_type}_pos": pos,
                                }
                            ]
                        )
                        token_match_found = True
                        if self.config.intervention_types[0] == SubloreftIntervention:
                            subspaces.append(self.word_to_subspace[word_matches[0]])
                        break
                if (
                    not token_match_found
                ):  # this could happen if the first token has some special character
                    for word_tokens in self.vocab[word_matches[0]]:
                        last_tok_match = torch.where(
                            tok_seq == torch.tensor(word_tokens)[-1]
                        )[0]
                        if last_tok_match.numel() > 0:
                            last_pos = last_tok_match[0].item()
                            for i in reversed(range(0, last_pos, 1)):
                                if word_matches[0] in self.tokenizer.decode(
                                    tok_seq[i : last_pos + 1],
                                    skip_special_tokens=True,
                                ):
                                    if self.pos_type == "last_token":
                                        pos = [last_pos]
                                    elif self.pos_type == "all_tokens":
                                        pos = list(range(i, last_pos + 1))
                                    else:
                                        raise ValueError(
                                            f"Invalid pos_type: {self.pos_type}"
                                        )
                                    result.append(
                                        [
                                            {
                                                "word": word_matches[0],
                                                f"{self.pos_type}_pos": pos,  # needs adaptation for all tokens
                                            }
                                        ]
                                    )
                                    token_match_found = True
                                    if self.config.intervention_types[0] == SubloreftIntervention:
                                        subspaces.append(self.word_to_subspace[word_matches[0]])
                                    break
                            if token_match_found:
                                break
                    if not token_match_found:
                        result.append(None)
                        if self.config.intervention_types[0] == SubloreftIntervention:
                            subspaces.append(None) # default subspace, will be purged by intervention mask anyway

        intervention_locs = [
            [loc_info[f"{self.pos_type}_pos"] for loc_info in r]
            if r is not None
            else [[0]]
            for r in result
            for _ in range(len(self.interventions))
        ]
        intervention_mask = torch.tensor(
            [1 if r is not None else 0 for r in result],
            device=tok_sequences.device,
        )
        if self.config.intervention_types[0] == SubloreftIntervention:
            subspaces = [[sub] for sub in subspaces for _ in range(len(self.interventions))]
        else:
            subspaces = None
        return {"sources->base": (None, intervention_locs)}, intervention_mask, subspaces


class SubloreftIntervention(NoreftIntervention):
    """
    This is a LoReFT that supports subspace interventions!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, base, source=None, subspaces=None):
        assert subspaces is not None
        output = []

        proj_base = self.proj_layer(base)
        diff = self.act_fn(self.learned_source(base)) - proj_base

        subspace_idx = (
            torch.tensor(subspaces)
            .repeat_interleave(base.shape[1], dim=0)
            .reshape(base.shape[0], base.shape[1], len(subspaces[0]))
            .to(base.device)
        )
        w_idx = (
            torch.tensor(subspaces)
            .repeat_interleave(base.shape[-1], dim=1)
            .reshape(base.shape[0], len(subspaces[0]), base.shape[-1])
            .to(base.device)
        )
        batched_subspace = diff.gather(dim=-1, index=subspace_idx)
        batched_weights = self.proj_layer.weight[None, :].repeat_interleave(len(subspaces), dim=0).gather(1, w_idx)        
        
        batched_output = torch.bmm(batched_subspace, batched_weights)
        output = base + batched_output

        return self.dropout(output.to(base.dtype))


def get_babelreft_model(
    model,
    reft_config,
    pos_type,
    low_rank_dim,
    tokenizer,
    words_to_add=None,
    set_device=True,
    disable_model_grads=True,
):
    """
    Create an instance of BabelReFT model.
    """
    reft_model = BabelReftModel(reft_config, model, pos_type, tokenizer, words_to_add, low_rank_dim)
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
    elif hparams.intervention_type == "subloreft":
        intervention = SubloreftIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=hparams.low_rank_dim * hparams.num_edits,
            add_bias=False,
        )
        low_rank_dim = hparams.low_rank_dim * hparams.num_edits
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
