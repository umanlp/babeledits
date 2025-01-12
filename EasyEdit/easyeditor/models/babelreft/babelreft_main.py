from typing import Dict, List, Optional
import numpy as np
from pyreft import ReftModel
import torch
from .babelreft_hparams import BabelReFTHyperParams
import pyreft
from ...util import nethook
import datasets
from pyreft.dataset import ReftDataCollator
from pyreft import ReftTrainerForCausalLM
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
    DataCollatorForSeq2Seq,
)
import copy as cp


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

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": full_input_ids,
            "intervention_locations": torch.tensor(
                model.get_unit_locations(full_input_ids)[0]["sources->base"][1]
            )
            .permute(1, 0, 2)
            .tolist(),
            "labels": target_ids,
        }
    )

    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    data_module = dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

    train_module = data_module
    training_args = TrainingArguments(
        num_train_epochs=hparams.num_steps,
        output_dir="./reft_res/",
        per_device_train_batch_size=1,
        learning_rate=hparams.lr,
        logging_steps=1,
        report_to=[],
    )
    trainer = ReftTrainerForCausalLM(
        model=model,
        tokenizer=tok,
        args=training_args,
        callbacks=[TrainingLossThresholdCallback(threshold=0.01)],
        **train_module,
    )

    _ = trainer.train()

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
        *args,
        **kwargs,
    ):
        super(BabelReftModel, self).__init__(reft_config, model)
        self.vocab = []
        self.tokenizer = tokenizer
        self.pos_type = pos_type
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
            unit_locations, intervention_mask = self.get_unit_locations(
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
                        _, intv_output = super().forward(
                            base=sel_base,
                            output_original_output=False,
                            unit_locations=sel_unit_locations,
                        )
                        all_logits.append(intv_output.logits.bfloat16())
                    else:
                        all_logits.append(og_output.logits[i].unsqueeze(0).bfloat16())
                output_logits = torch.cat(all_logits, dim=0)
            return output_logits

    def add_words_to_vocab(self, word_list):
        exp_word_list = [[word, f" {word}"] for word in word_list]
        exp_word_list = [
            item
            for sublist in exp_word_list
            for item in sublist
            if item not in self.vocab
        ]
        if len(exp_word_list) == 0:
            return
        token_seqs = self.tokenizer(exp_word_list, add_special_tokens=False)[
            "input_ids"
        ]
        self.vocab += token_seqs

    def get_vocab_words(self):
        return self.tokenizer.batch_decode(self.vocab)

    def reset_vocab(self):
        self.vocab.clear()

    def get_unit_locations(self, tok_sequences):
        if self.pos_type == "last_token":
            result = []
            for seq in tok_sequences:
                # print(seq, tokenizer.decode(seq))
                d = []
                np_seq = seq.cpu().numpy()
                for idx, word_tokens in enumerate(self.vocab):
                    word_len = len(word_tokens)
                    matches = (np_seq[np.arange(len(seq) - word_len + 1)[:, None] + np.arange(word_len)] == word_tokens).all(axis=1)
                    if matches.any():
                        d.append(
                            {
                                "word_idx": idx,
                                "last_token_pos": matches.argmax().item() + word_len - 1,
                            }
                        )
                        break
                result.append(d or None)
            intervention_locs = [
                [[loc_info["last_token_pos"]] for loc_info in r]
                if r is not None
                else [[0]]
                for r in result
                for _ in range(len(self.interventions))
            ]
            intervention_mask = torch.tensor(
                [
                    1 if r is not None else 0
                    for r in result
                ],
                device=tok_sequences.device,
            )
            return {
                "sources->base": (None, intervention_locs)
            }, intervention_mask
        elif self.pos_type == "all_tokens":
            result = []
            for seq in tok_sequences:
                # print(seq, tokenizer.decode(seq))
                d = []
                np_seq = seq.cpu().numpy()
                for idx, word_tokens in enumerate(self.vocab):
                    word_len = len(word_tokens)
                    matches = (
                        np_seq[
                            np.arange(len(seq) - word_len + 1)[:, None]
                            + np.arange(word_len)
                        ]
                        == word_tokens
                    ).all(axis=1)
                    if matches.any():
                        start_idx = matches.argmax().item()
                        d.append(
                            {
                                "word_idx": idx,
                                "all_token_pos": list(range(start_idx, start_idx + word_len)),
                            }
                        )
                        break
                result.append(d or None)
            intervention_locs = [
                [loc_info["all_token_pos"] for loc_info in r] if r is not None else [[0]]
                for r in result for _ in range(len(self.interventions))
            ]
            intervention_mask = torch.tensor(
                [1 if r is not None else 0 for r in result], device=tok_sequences.device
            )
            return {"sources->base": (None, intervention_locs)}, intervention_mask
        else:
            raise ValueError(f"Invalid pos_type: {self.pos_type}")


def get_babelreft_model(
    model,
    reft_config,
    pos_type,
    tokenizer,
    words_to_add=None,
    set_device=True,
    disable_model_grads=True,
):
    """
    Create an instance of BabelReFT model.
    """
    reft_model = BabelReftModel(reft_config, model, pos_type, tokenizer, words_to_add)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()
    return reft_model


def get_reft_config(hparams, hidden_size):
    reft_cfg = pyreft.ReftConfig(
        representations=[
            {
                "layer": layer,  # only single layer intervention supported
                "component": "block_output",
                "unit": "pos",
                "low_rank_dimension": hparams.low_rank_dim,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=hidden_size,
                    low_rank_dimension=hparams.low_rank_dim,
                ),
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
