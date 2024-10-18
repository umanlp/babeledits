from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np

def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]
        
def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys
    
def summary_metrics(all_metrics, eval_metrics, locality_metrics, rewrite_metrics=None):
    rewrite_metrics = rewrite_metrics or eval_metrics
    if isinstance(all_metrics, dict):
        all_metrics = [all_metrics, ]
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    output_file = os.path.join(logs_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    mean_metrics = dict()
    for eval in ["pre", "post"]:
        mean_metrics[eval] = dict()
        for key in ["rewrite_acc", "rewrite_ppl"]:
            if key in all_metrics[0][eval].keys():
                mean_metrics[eval][key] = dict()
            else:
                continue
            for metric_type in rewrite_metrics:
                mean_metrics[eval][key].update({metric_type : np.mean([np.max(score[eval][key][metric_type]) for score in all_metrics])})
        if "ppl" in all_metrics[0][eval].keys():
            mean_metrics[eval]["ppl"] = dict()
            for lang in all_metrics[0][eval]["ppl"]:
                mean_metrics[eval]["ppl"][lang] = np.mean([score[eval]["ppl"][lang] for score in all_metrics])
        for key in ["rephrase_acc", "locality", "portability"]:
                mean_metrics[eval][key] = dict()
                if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                    for prompt_type in all_metrics[0][eval][key]:
                        mean_metrics[eval][key][prompt_type] = dict()
                        metrics_to_gather = eval_metrics if key != "locality" else locality_metrics
                        for metric_type in metrics_to_gather:
                            mean_metrics[eval][key][prompt_type].update({metric_type: np.mean([np.max(score[eval][key][prompt_type][metric_type]) for score in all_metrics if prompt_type in score[eval][key]])})

                    # for lkey in get_all_acc_keys(all_metrics):
                    #     metrics = [metric[eval][metric_type][key][lkey] for metric in all_metrics if lkey in metric[eval][metric_type][key].keys()]
                    #     if len(metrics) > 0:
                    #         mean_metrics[eval][metric_type][key][lkey] = np.mean(metrics)
                        # mean_metrics[eval][key][lkey] = np.mean(
                        #     [metric[eval][key][lkey] for metric in all_metrics])
    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

    return mean_metrics

def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Union[str, List[str]],
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_,
        'rephrase_prompt': {},
        'portability': {},
        'locality': {}
    }
    for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
    ]

    if 'aliases' in kwargs:
        aliases = kwargs['aliases']
        for idx, req in enumerate(requests):
            if kwargs['edit_lang'] in aliases['rel_aliases'][idx]:
                req['aliases'] = aliases['rel_aliases'][idx][kwargs['edit_lang']]

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        else:
            assert len(kwargs["loc_prompts"]) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        for generality_key in rephrase_prompts.keys():
            for i, request in enumerate(requests):
                if rephrase_prompts[generality_key]["prompt"][i] is not None:
                    request["rephrase_prompt"].update(
                        {
                            generality_key: {
                                "prompt": rephrase_prompts[generality_key]["prompt"][i],
                                "ground_truth": rephrase_prompts[generality_key]["ground_truth"][i]
                            }
                        }
                    )
                    if 'aliases' in kwargs and 'gen_aliases' in aliases:
                        tgt_lang = generality_key[-2:]
                        if tgt_lang in aliases['gen_aliases'][i]:
                            request['rephrase_prompt'][generality_key]['ground_truth'] = [request['rephrase_prompt'][generality_key]['ground_truth'], *aliases['gen_aliases'][i][tgt_lang]]
                            request['rephrase_prompt'][generality_key]['prompt'] = [request['rephrase_prompt'][generality_key]['prompt']]*(len(request['rephrase_prompt'][generality_key]['ground_truth']))

    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            == len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            == len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
                    if 'aliases' in kwargs and any([port_type in aliases for port_type in ['xlt_port_aliases', 'multi-hop_aliases', 'subj_aliases']]):
                        if "xlt" in portability_key:
                            tgt_lang = portability_key[-2:]
                            if tgt_lang in aliases['xlt_aliases'][i]:
                                request['portability'][portability_key]['ground_truth'] = [request['portability'][portability_key]['ground_truth'], *aliases['xlt_aliases'][i][tgt_lang]]
                        if "multi-hop" in portability_key:
                            tgt_lang = portability_key[-2:]
                            if tgt_lang in aliases['multi-hop_aliases'][i]:
                                request['portability'][portability_key]['ground_truth'] = [request['portability'][portability_key]['ground_truth'], *aliases['multi-hop_aliases'][i][tgt_lang]]
                        if "subj" in portability_key:
                            tgt_lang = portability_key[-2:]
                            if tgt_lang in aliases['subj_aliases'][i]:
                                request['portability'][portability_key]['ground_truth'] = [request['portability'][portability_key]['ground_truth'], *aliases['subj_aliases'][i][tgt_lang]]
                        
                        request['portability'][portability_key]['prompt'] = [request['portability'][portability_key]['prompt']]*len(request['portability'][portability_key]['ground_truth'])
                        
    return requests
