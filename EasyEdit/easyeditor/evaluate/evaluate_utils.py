import torch
import math
import numpy as np
import scipy
import nltk
import typing
from ..util.generate import generate_fast
import torch.nn.functional as F
from ..trainer import *
from sklearn.metrics import f1_score
import openai
from evaluate import logging
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        targets,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            answers = ans.squeeze().detach().cpu().numpy().tolist()
            return answers if type(answers[0]) is list else [answers,]
        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()

def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=False, eval_metric="token_em", generation_conf=None):
    assert not model.training
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(target_new, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(device)
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                use_cache=False,
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results
    
    if eval_metric == "token_em":
        if isinstance(prompts, str):
            prompts,targets = [prompts,], [targets,]
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
        max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        prompt_target_tok = tok(
            prompt_target,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        ).to(f"cuda:{device}")
        prompt_tok = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        )
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
        with torch.no_grad():
            outputs = model(**prompt_target_tok)
            if type(outputs) is torch.Tensor:
                logits = outputs
            else:
                logits = outputs.logits
            answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
            labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
            answers = slice_list(answers,prompt_len,left=True)
            labels = slice_list(labels,prompt_len,left=False)
            if locality:
                return answers if type(answers[0]) is list else [answers,]
            if isinstance(answers[0], list):
                res = []
                for ans,label in zip(answers,labels):
                    temp_acc = float(np.mean(np.equal(ans, label)))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                return res
            else:
                return [float(np.mean(np.equal(answers, labels)))]
            
    elif eval_metric == "token_em_lm_eval":
        if isinstance(prompts, str):
            prompts,targets = [prompts,], [targets,]
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
        max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        prompt_target_tok = tok(
            prompt_target,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        ).to(f"cuda:{device}")
        prompt_tok = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        )
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
        if model.prev_logits is not None:
            outputs = model.prev_logits
        else:
            with torch.no_grad():
                outputs = model(**prompt_target_tok)
                model.prev_logits = outputs
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = float(np.all(np.equal(ans, label)))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [float(np.all(np.equal(answers, labels)))]
            
    elif eval_metric == "rewrite_score":
        if isinstance(prompts, str):
            prompts,targets = [prompts,], [targets,]
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
        max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        prompt_target_tok = tok(
            prompt_target,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        ).to(f"cuda:{device}")
        prompt_tok = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        )
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
        if model.prev_logits is not None:
            outputs = model.prev_logits
        else:
            with torch.no_grad():
                outputs = model(**prompt_target_tok)
                model.prev_logits = outputs 
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
            
        shifted_logits = logits[:, :-1, :]
        shifted_labels = prompt_target_tok["input_ids"][:, 1:]
        target_mask = []
        for l in prompt_len:
            target_mask.append([0]*(l-1) + [1]*(shifted_labels.shape[-1]-l+1))
        target_mask = torch.tensor(target_mask).to(logits.device)
        log_probs = gather_log_probs(shifted_logits, shifted_labels) * target_mask
        probs = log_probs.sum(1).exp().float().detach().cpu().numpy().tolist()

        return probs
    
    elif eval_metric == "first_token_em":
        if isinstance(prompts, str):
            prompts,targets = [prompts,], [targets,]
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
        max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        prompt_tok = tok(
            prompts, # A single repeated prompt
            padding=True,
            truncation=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        ).to(f"cuda:{device}")
        prompt_target_tok = tok(
            prompt_target,
            padding=True,
            max_length=max(hparams.max_length, max_prompt_len),
            return_tensors="pt",
        ).to(f"cuda:{device}")
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
        prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
        if len(set(prompts)) == 1 and len(prompts) > 1:
            # Single repeated prompt
            input_tok = {"input_ids": prompt_tok["input_ids"][None, 0], "attention_mask": prompt_tok["attention_mask"][None, 0]}
        else:
            input_tok = prompt_tok
        if model.prev_logits is not None:
            outputs = model.prev_logits
        else:
            with torch.no_grad():
                outputs = model(**prompt_target_tok)
                model.prev_logits = outputs
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        last_non_masked = input_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        answers = torch.argmax(gathered, dim=1).detach().cpu().numpy().tolist()
        if len(prompts) > 1 and len(set(prompts)) == 1:
            answers = answers * len(prompts)
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        labels = slice_list(labels,prompt_len,left=False) # get the answer in isolation
        if isinstance(labels[0], list):
            assert(prompts == [prompts[0]] * len(prompts))
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = float(np.mean(np.equal(ans, label[0])))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            assert len(prompts) == 1
            return [float(np.mean(np.equal(answers, labels[0])))]
    # elif eval_metric == "efficacy_magn":
        # if isinstance(prompts, str):
            # prompts,targets = [prompts,], [targets,]
        # prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
        # max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
        # prompt_target_tok = tok(
            # prompt_target,
            # padding=True,
            # truncation=True,
            # max_length=max(hparams.max_length, max_prompt_len),
            # return_tensors="pt",
        # ).to(f"cuda:{device}")
        # prompt_tok = tok(
            # prompts,
            # padding=True,
            # truncation=True,
            # max_length=max(hparams.max_length, max_prompt_len),
            # return_tensors="pt",
        # )
        # num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
        # num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
        # prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
        # with torch.no_grad():
            # outputs = model(**prompt_target_tok)
            # if type(outputs) is torch.Tensor:
                # logits = outputs
            # else:
                # logits = outputs.logits
        # label_tokens = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        # label_tokens = slice_list(label_tokens,prompt_len,left=False)
        # log_probs = logits.log_softmax(dim=-1)
        # res = []
        # for idx in range(len(prompts)):
            # p_len = prompt_len[idx]
            # mask = torch.tensor([[p_len - 1 + pos_idx, vocab_idx] for pos_idx, vocab_idx in enumerate(label_tokens)], device=log_probs.device)
            # sel_log_probs = log_probs[idx, mask[:, 0], mask[:, 1]]
            # res.append({"log_probs" : sel_log_probs})
        # return res
    elif eval_metric == "string_em":        
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        max_prompt_len = max([len(tok.encode(_)) for _ in prompts]) + 1

        if len(set(prompts)) == 1 and len(prompts) > 1: # A single repeated prompt
            prompt_tok = tok(
                prompts[0] ,
                padding=True,
                truncation=True,
                max_length=max(hparams.max_length, max_prompt_len),
                return_special_tokens_mask=True,
                return_tensors="pt",
            ).to(f"cuda:{device}")
            generation_conf.max_new_tokens = prompt_tok.input_ids.shape[1] - prompt_tok["special_tokens_mask"].sum().item()

            full_output = generate_answer(model, prompt_tok, tok, generation_conf)
            generated_output = full_output[len(prompts[0]):].strip()

            for target_new in targets:
                # Check for each answer in the generated output
                if target_new in generated_output:
                    results.append(1.0)
                else:
                    results.append(0.0)
        else:
            for prompt, target_new in zip(prompts, targets):
                # Tokenize and preprocess the prompt
                prompt_tok = tok(
                    prompt,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                    truncation=True,
                    max_length=max(hparams.max_length, max_prompt_len)
                ).to(device)

                generation_conf.max_new_tokens = prompt_tok.input_ids.shape[1] - prompt_tok["special_tokens_mask"].sum().item()

                full_output = generate_answer(model, prompt_tok, tok, generation_conf)
                generated_output = full_output[len(prompt):].strip()

                # Check for each answer in the generated output
                if target_new in generated_output:
                    results.append(1.0)
                else:
                    results.append(0.0)
        return results
    else:
        raise ValueError(f"Invalid eval_metric: {eval_metric}")

def generate_answer(model, prompt_tok, tok, generation_conf):
    # Generate output from the model
    with torch.no_grad():
        outputs = model.generate(input_ids=prompt_tok['input_ids'], attention_mask=prompt_tok['attention_mask'], generation_config=generation_conf, pad_token_id=tok.eos_token_id)

    # Decode the output
    full_output = tok.decode(outputs[0], skip_special_tokens=True)
    
    return full_output

def test_generation_quality_serac(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,       
):
    #only single case
    prompt_tok = tok(
        prefixes,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    prompt_tok_length=len(prompt_tok['input_ids'])
    gen_texts=model.generate(**prompt_tok,max_new_tokens=256)
    if isinstance(model,SERAC):
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))
    else:
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))      
    ngram_entropy = n_gram_entropy(gen_texts, return_list=True)


    ret = {
        "ngram_entropy": ngram_entropy
    }
    return ret

def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    vanilla_generation: bool = False,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
        vanilla_generation=vanilla_generation,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    ret = {
        "ngram_entropy": ngram_entropy,
    }
    return ret

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

def PPL(
    model,
    tok,
    prompt: typing.Union[str, typing.List[str]],
    target_new: typing.Union[str, typing.List[str]],
    device,
):
    if isinstance(prompt, str):
        prompt,target_new = [prompt,], [target_new,]
    full_prompt = [f"{p} {l}" for p, l in zip(prompt, target_new)]
    prompt_ids = tok(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][:num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100 # What is this doing?
    batch = {f"{k1}" : v1 for k1, v1 in tokens.items()}
    input_ids = batch["input_ids"][:, :1024]#.to(device)
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=target_ids.to(device))
        nll = outputs.loss
    ppl = torch.exp(nll)#.clip(0, 100)
    return ppl.cpu().numpy().tolist()

def verify_answer(model_answer, correct_answer):
    if type(correct_answer) is str:
        correct_answer = [[correct_answer]]
    for answer in correct_answer:
        if True not in [possible_answer in model_answer for possible_answer in answer]:
            return False
    return True

def answer_match(
    model,
    tok,
    prompt: str,
    target_new: str,
    device,
):
    inputs = tok.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, temperature=0, max_new_tokens=30)
    predict = tok.decode(outputs[0], skip_special_tokens=True)

    return verify_answer(predict,target_new)

def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

def gather_log_probs(logits, labels):
    # print(f"labels.shape: {labels.shape} , logits.shape[:-1] :{logits.shape[:-1]}")
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def es(pre_logits, edit_logits, q_mask, labels, same_mask):
    
    _, targ = mask_hf_labels(labels)

    pos_mask = same_mask.unsqueeze(-1) * q_mask 
    neg_mask = (~same_mask).unsqueeze(-1) * q_mask 
        
    pre_token_log_probs = gather_log_probs(pre_logits, targ)
    edit_token_log_probs = gather_log_probs(edit_logits, targ)

    mean_pos_pre = masked_mean(pre_token_log_probs, pos_mask)
    mean_pos_edit = masked_mean(edit_token_log_probs, pos_mask)
    mean_neg_edit = masked_mean(edit_token_log_probs, neg_mask)

    z_sent = (mean_pos_edit - mean_neg_edit).sigmoid()
    z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
    z_topic = min(1, z_topic_raw)

    es_sent = z_sent * z_topic
    return es_sent

def es_per_icl(example, pre_logits, edit_logits):
    with torch.no_grad():
        
        pre_q_mask = example["outer_pre"]["q_mask"]
        edit_q_mask = example["outer_edit"]["q_mask"]
        
        pre_labels = example["outer_pre"]["labels"]
        edit_labels = example["outer_edit"]["labels"]
        
        pre_mask, pre_targ = mask_hf_labels(pre_labels)
        edit_mask, edit_targ = mask_hf_labels(edit_labels)
        
        same_per_mask = example["same_per_mask"]

        pre_pos_mask = same_per_mask.unsqueeze(-1) * pre_q_mask 
        pre_neg_mask = (~same_per_mask).unsqueeze(-1) * pre_q_mask 
        edit_pos_mask = same_per_mask.unsqueeze(-1) * edit_q_mask 
        edit_neg_mask = (~same_per_mask).unsqueeze(-1) * edit_q_mask 
        
        pre_token_log_probs = gather_log_probs(pre_logits, pre_targ)
        edit_token_log_probs = gather_log_probs(edit_logits, edit_targ)

        mean_pos_pre = masked_mean(pre_token_log_probs, pre_pos_mask)
        mean_pos_edit = masked_mean(edit_token_log_probs, edit_pos_mask)
        mean_neg_edit = masked_mean(edit_token_log_probs, edit_neg_mask)

        z_per = (mean_pos_edit - mean_neg_edit).sigmoid()
        z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_edit,
            "wrong_probs": mean_neg_edit,
        }

def per_generation(
    model,
    tok,
    max_out_len: int,
    target_per, 
    device,
    edited_model=None,
    IKE=False,
    **kwargs
    ):
    def generate_text(query, model, tokenizer):
        input_text = query
        generation_config = {
            "max_new_tokens": max_out_len,
            "temperature": 0,
            "eos_token_id": tokenizer.eos_token_id,
        }
        src_input_ids = tokenizer(input_text).input_ids
        input_ids = torch.tensor([src_input_ids], dtype=torch.long, device=device)
        outputs = model.generate(input_ids, **generation_config)
        response = tokenizer.decode(outputs[0][len(src_input_ids) :], skip_special_tokens=True)
        return response
    
    def clean_text(text):
        return text.strip().split("\n")[0]
    
    if IKE:
        pre_text = clean_text(generate_text(kwargs["pre_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["edit_q"], model, tok))

    else:
        assert edited_model is not None
        pre_text = clean_text(generate_text(kwargs["inner_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["inner_q"], edited_model.model, tok))

    ngram_pre_text = n_gram_entropy([pre_text])
    ngram_edit_text = n_gram_entropy([edit_text])
    coherent = ngram_pre_text >= 3.5 and ngram_edit_text >= 3.5
    
    result = {
        "pre_text": pre_text,
        "edit_text": edit_text,
        "ngram_pre_text": ngram_pre_text,
        "ngram_edit_text": ngram_edit_text,
        "coherent": coherent,
        "target_per": target_per,
    }

    return result

def kl_loc_loss(pre, post, mask=None):
    
    pre = pre.to(torch.float32).contiguous()
    post = post[:,-pre.shape[1]:,:].to(torch.float32).contiguous()
    
    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        # print("sequence")
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError

def F1(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=True):
    if vanilla_generation:
        target_new_tokens = tok.encode(targets, add_special_tokens=False)
        prompt_tok = tok(
            prompts,
            return_tensors="pt",
        ).to(device)
        gen_token = model.generate(
            input_ids=prompt_tok['input_ids'],
            attention_mask=prompt_tok['attention_mask'],
            max_new_tokens=len(target_new_tokens),
            pad_token_id=tok.eos_token_id,
            use_cache=False,

        )
        return f1_score(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):], average='macro')
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)

        return f1_score(answers, labels, average='macro')

def test_instance_change(model, tok, max_length, prompts, targets, device, P = None):
    demo1_str = "Whether FrancoAngeli belongs to category publisher? Yes\nWhether And Other Stories belongs to category people? No\n"
    if P is None:
        prompts = demo1_str +prompts
    else:
        prompts = P + demo1_str + prompts

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),
            max_new_tokens=2,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][model_response[0].rfind('?')+2:]
        # print(model_response[0], answer)

        if "yes" in answer.lower():
            return np.ones(1)
        else:
            if "no" not in answer.lower():
                print(f"entity error in define yes or no: {answer}")
                return np.array([-1.0])
            return np.zeros(1)

def test_concept_gen(model, tok, max_length, prompts, targets, device):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompts = [prompt + ' ' for prompt in prompts]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),
            max_new_tokens=40,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][len(prompts[0]):]
        return answer


def test_safety_gen(
        model, 
        tokenizer, 
        test_prompt, 
        cuda,
        max_tokens = 1624, 
        max_output_tokens=600):
    tokenizer.padding_side = 'left'
    # if input_tokens (at least 1024) + output_tokens (at least 600) < 1624, truncate the input length (from right to left, as harmful questions typically appear on the right)
    if max_tokens < 1624:
        only_response = []
        for item in test_prompt:
            input = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{cuda}")
            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                input = {k: v[:, -(max_tokens - max_output_tokens):] for k, v in input.items()}
            with torch.no_grad():
                outputs = model.generate(**input, max_new_tokens=max_output_tokens)
                texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                texts = texts[0]
            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                max_overlap_len = min(len(item), len(texts))
                overlap = next((item[-i:] for i in range(max_overlap_len, 0, -1) if item[-i:] == texts[:i]), "")
            else:
                overlap = item
            only_response.append(texts[len(overlap)+1:].lstrip())
        return only_response
    else:
        input = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{cuda}")
        with torch.no_grad():
            outputs = model.generate(**input, max_new_tokens=max_output_tokens)
            texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
        return only_response

def compute_ppl(
    predictions,
    model,
    model_name,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    """device has been set
    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:  # Dont go
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:  # Dont go
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous() #because we do not have the label for what comes next
        shift_labels = labels[..., 1:].contiguous()  # because we can predict only from the second token onwards
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()  # same reason as above

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def compute_bpb(
    predictions,
    model,
    model_name,
    lang_mask,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    """device has been set
    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:  # Dont go
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:  # Dont go
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    byte_lenghts = torch.tensor([len(s.encode("utf-8")) for s in predictions], device=device)

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    losses = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous() #because we do not have the label for what comes next
        shift_labels = labels[..., 1:].contiguous()  # because we can predict only from the second token onwards
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()  # same reason as above

        loss_sum = (
            loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1) 
        losses += loss_sum.tolist()
    
    losses = torch.tensor(losses)
    lang_to_bpb = {}
    for lang in lang_mask:
        losses_lang = losses[lang_mask[lang]].sum()
        bpb_lang = (1 / (byte_lenghts[lang_mask[lang]].sum() *  math.log(2))) * losses_lang 
        # Alternative formulation
        # ppl = torch.exp(1/(attn_masks[lang_mask[lang]].sum()) * losses[lang_mask[lang]].sum())
        # token_to_byte_ratio = (attn_masks[lang_mask[lang]].sum() / byte_lenghts[lang_mask[lang]].sum())
        # bpb_lang2 = token_to_byte_ratio* torch.log(ppl) / math.log(2)
        lang_to_bpb[lang] = float(bpb_lang.item())
        

    return lang_to_bpb
