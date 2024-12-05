from pyexpat import model
from typing import Optional, Union, List, Tuple, Dict
from time import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality
from ..evaluate.evaluate import compute_locality_quality
from ..evaluate.evaluate_utils import compute_ppl, compute_bpb
from pathlib import Path
import copy
import gzip

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if hasattr(hparams,'fp16') and hparams.fp16:
                torch_dtype = torch.float16
            elif hasattr(hparams,'bf16') and hparams.bf16:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            print(f">> Using torch_dtype: {torch_dtype} ++")
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'bloom' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
            elif 'aya' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if "Causal" in self.model.__class__.__name__:
                self.tok.padding_side = 'left'
            # if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                # LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                # self.tok.padding_side = 'left'
            # if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                # LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                # self.tok.padding_side = 'right'
            
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel: 
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        model.prev_logits = None
        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             return_edited_weights=False,
             eval_metrics=["token_em"],
             generation_conf=None,
             locality_metrics=["token_em"],
             lm_cfg=None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:# Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, 
                                return_edited_weights=return_edited_weights, eval_metrics=eval_metrics, generation_conf=generation_conf, locality_metrics=locality_metrics, lm_cfg=lm_cfg, **kwargs)

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_prompts: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   verbose=True,
                   eval_metrics=["token_em"],
                   generation_conf=None,
                   locality_metrics=["token_em"],
                   ppl_cfg = None,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                          locality_prompts, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        
        if ppl_cfg: # TODO adapt to lm_cfg with bpb
            ppl_output = compute_ppl(ppl_cfg['prompts'], self.model, self.model_name, batch_size=ppl_cfg['batch_size'], device=self.hparams.device, add_start_token=True)['perplexities']
            print(f"Perplexities before editing: {ppl_per_lang}")

        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metrics=eval_metrics, test_generation=test_generation, generation_conf=generation_conf, locality_metrics=locality_metrics),
                }

                if ppl_cfg:
                    ppl_output = compute_ppl(ppl_cfg['prompts'], edited_model, self.model_name, batch_size=ppl_cfg['batch_size'], device=self.hparams.device, add_start_token=True)['perplexities']
                    ppl_per_lang = {lang:float(np.mean([ppl_output[idx + j*len(ppl_cfg["langs"])] for j in range(ppl_cfg["num_sent_per_lang"])]) for idx, lang in enumerate(ppl_cfg["langs"]))}
                    metrics['post'].update({"ppl":ppl_per_lang})
                
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metrics=eval_metrics, test_generation=test_generation, generation_conf=generation_conf, locality_metrics=locality_metrics)

                if ppl_cfg:
                    ppl_per_lang = {lang:float(np.mean([ppl_output[idx + j*len(ppl_cfg["langs"])] for j in range(ppl_cfg["num_sent_per_lang"])]) for idx, lang in enumerate(ppl_cfg["langs"]))}
                    chunk_metrics[i]['pre'].update({"ppl":ppl_per_lang})
                
                if 'locality' in chunk_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        for loc_metric in locality_metrics:
                            locality_result = []
                            output_type = "output" if loc_metric == "token_em" else "logprobs"
                            for loc_pre, loc_post in zip(chunk_metrics[i]['pre']['locality'][locality_key][loc_metric][output_type], chunk_metrics[i]['post']['locality'][locality_key][loc_metric][output_type]):
                                if loc_metric == "token_em":
                                    locality_result.append(np.mean(np.equal(loc_pre, loc_post)))
                                elif loc_metric == "nkl":
                                    locality_result.append(F.kl_div(input=loc_post, target=loc_pre, reduction='sum', log_target=True).item())
                            chunk_metrics[i]['post']['locality'][locality_key][loc_metric].pop(output_type)
                            chunk_metrics[i]['post']['locality'][locality_key][loc_metric] = locality_result

                    chunk_metrics[i]['pre'].pop('locality')

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             return_edited_weights=False,
             eval_metrics=["token_em"],
             generation_conf=None,
             locality_metrics=["token_em"],
             lm_cfg = None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        # eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                metrics= {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metrics=eval_metrics, test_generation=test_generation, generation_conf=generation_conf, locality_metrics=locality_metrics)}
                all_metrics.append(metrics)

            if lm_cfg:
                lm_score = self.evaluate_language_modeling(lm_cfg)
                for evaluation in all_metrics:
                    evaluation["pre"].update({lm_cfg['metric']: lm_score})
                print(f"Language Modeling Score(s) using {lm_cfg['metric']} : {lm_score}")

            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                print(f"Saving pre-edit metrics to {kwargs['pre_file']}")
                Path(kwargs['pre_file']).parent.mkdir(parents=True, exist_ok=True)
                copy_metrics = copy.deepcopy(all_metrics)
                for evaluation in copy_metrics:
                    for loc_key in evaluation['pre']['locality']:
                        # serializing tensors
                        evaluation['pre']['locality'][loc_key]['nkl']['logprobs'] = evaluation['pre']['locality'][loc_key]['nkl']['logprobs'].tolist()
                with gzip.open(kwargs['pre_file'], 'wt') as f:
                    json.dump(copy_metrics, f)
                if 'pre_eval_only' in kwargs.keys() and kwargs['pre_eval_only']:
                    return copy_metrics, None, None

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, eval_metrics, generation_conf, **kwargs):
            all_metrics[idx].update({ 
                'case_id': idx,
                "requested_rewrite": request,
                "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metrics=eval_metrics, test_generation=test_generation, generation_conf=generation_conf, locality_metrics=locality_metrics, icl_examples=icl_examples, icl_template="{icl_examples}{prompt} {target}" if self.alg_name == "IKE" else None),
            })
            #TODO continue
            if "metric_kwargs" in kwargs:
                all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))

            if 'locality' in all_metrics[idx]['post'].keys():
                for locality_key in request['locality'].keys():
                    for loc_metric in locality_metrics:
                        locality_result = []
                        output_type = "output" if loc_metric == "token_em" else "logprobs"
                        for loc_pre, loc_post in zip(all_metrics[idx]['pre']['locality'][locality_key][loc_metric][output_type], all_metrics[idx]['post']['locality'][locality_key][loc_metric][output_type]):
                            if loc_metric == "token_em":
                                locality_result.append(float(np.mean(np.equal(loc_pre, loc_post))))
                            elif loc_metric == "nkl":
                                locality_result.append(F.kl_div(input=loc_post, target=loc_pre, reduction='sum', log_target=True).item())
                        all_metrics[idx]['post']['locality'][locality_key][loc_metric].pop(output_type)
                        all_metrics[idx]['post']['locality'][locality_key][loc_metric] = locality_result

                all_metrics[idx]['pre'].pop('locality')

            if "rewrite_score" in all_metrics[idx]['pre']['rewrite_acc']:
                
                # Reliability
                p_pre_list = all_metrics[idx]['pre']['rewrite_acc']['rewrite_score']
                p_post_list = all_metrics[idx]['post']['rewrite_acc']['rewrite_score']
                all_metrics[idx]["post"]["rewrite_acc"]["rewrite_score"] = [
                    (p_post - p_pre) / (1 - p_pre)
                    for p_post, p_pre in zip(p_post_list, p_pre_list)
                ] 
                all_metrics[idx]['pre']['rewrite_acc'].pop('rewrite_score')

                # Portability and Generality
                for aspect in [x for x in ["rephrase_acc", "portability" ] if x in all_metrics[idx]['post'].keys()]:
                    for prompt_type in all_metrics[idx]['pre'][aspect]:
                        p_pre_list = all_metrics[idx]['pre'][aspect][prompt_type]['rewrite_score']
                        p_post_list = all_metrics[idx]['post'][aspect][prompt_type]['rewrite_score']
                        all_metrics[idx]['post'][aspect][prompt_type]['rewrite_score'] = [(p_post - p_pre)/(1 - p_pre) for p_post,p_pre in zip(p_post_list,p_pre_list)]
                        all_metrics[idx]['pre'][aspect][prompt_type].pop('rewrite_score')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")


        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                if return_edited_weights:
                    if i == 0:
                        weights_per_edit = {k: [] for k in weights_copy.keys()}
                    for k, v in weights_copy.items():
                        weights_per_edit[k].append(nethook.get_parameter(self.model, k).detach().clone().cpu())
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, eval_metrics, generation_conf, **kwargs)
                if lm_cfg:
                    lm_score = self.evaluate_language_modeling(lm_cfg)
                    all_metrics[i]['post'].update({lm_cfg['metric']:lm_score})
                    print(f"Language Modeling Score(s) using {lm_cfg['metric']} : {lm_score}")
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, eval_metrics, generation_conf, **kwargs)
                if lm_cfg:
                    lm_score = self.evaluate_language_modeling(lm_cfg)
                    all_metrics[i]['post'].update({lm_cfg['metric']:lm_score})
                    print(f"Language Modeling Score(s) using {lm_cfg['metric']} : {lm_score}")
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA':
                    self.model = edited_model
                else:
                    if return_edited_weights:
                        with torch.no_grad():
                            if i == 0:
                                weights_per_edit = {k: [] for k in weights_copy.keys()}
                            for k, v in weights_copy.items():
                                    weights_per_edit[k].append(nethook.get_parameter(self.model, k).detach().clone().cpu())
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(
                                f"cuda:{self.hparams.device}"
                            )

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            lm_metric = lm_cfg['metric'] if lm_cfg else None
            print(summary_metrics(all_metrics, eval_metrics, locality_metrics, lm_metric=lm_metric))

        if return_edited_weights:
            return all_metrics, edited_model, weights_copy, weights_per_edit
        else:
            return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy
    
    def generate_edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        locality_inputs:  Optional[Dict] = None,
        portability_inputs: Optional[Dict] = None,
        sequential_edit=False,
        verbose=True,
        **kwargs
    ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)

        assert len(prompts) == len(target_new)

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        
        def text_generate(
            model,
            model_name,
            hparams: HyperParams,
            tok: AutoTokenizer,
            query,
            device,
            eval_metric: str = 'token_em',
            test_generation = False
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=512
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            results = kwargs['pre_edit']
            all_results = results
        else:
            for i, request in enumerate(tqdm(requests)):
                results = {}
                results['pre'] = {}
                results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results['pre']['locality_ans'] = loc_results
                results['pre']['portability_ans'] = por_results
                all_results.append(results)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        
        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                results_post = {}
                results_post['rewrite_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results_post['rephrase_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results_post['locality_ans'] = loc_results
                results_post['portability_ans'] = por_results
                if test_generation:
                    if self.hparams.alg_name == 'GRACE':
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=True)
                    else:
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=False)
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": results_post
                })
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_results) != 0:
            summary_metrics(all_results)

        return all_results, edited_model, weights_copy

    def evaluate_language_modeling(self, lm_cfg):
        if lm_cfg['metric'] == 'ppl':
            ppl_output = compute_ppl(lm_cfg['prompts'], self.model, self.model_name, batch_size=lm_cfg['batch_size'], device=self.hparams.device, add_start_token=True)['perplexities']
            ppl_per_lang = {lang: float(np.mean([ppl_output[idx + j * len(lm_cfg["langs"])] for j in range(lm_cfg["num_sent_per_lang"])])) for idx, lang in enumerate(lm_cfg["langs"])}
            return ppl_per_lang
        if lm_cfg['metric'] == 'bpb':
            lang_mask = {lang: [idx + j * len(lm_cfg["langs"]) for j in range(lm_cfg["num_sent_per_lang"])] for idx, lang in enumerate(lm_cfg["langs"])}
            bpb_per_lang = compute_bpb(lm_cfg['prompts'], self.model, self.model_name, lang_mask, batch_size=lm_cfg['batch_size'], device=self.hparams.device, add_start_token=True)
            return bpb_per_lang