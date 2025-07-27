# Babeledits

This repository contains the code for building the BabelEdits dataset and for
evaluating various Knowledge Editing methods on it. It also contains code for
generating the dataset from scratch.

This code has been open-sourced alongside the following ACL'25 paper:

[**BABELEDITS: A Benchmark and a Modular Approach for Robust Cross-lingual Knowledge Editing of Large Language Models**](https://aclanthology.org/2025.findings-acl.438/), Tommaso Green, Félix Gaschi, Fabian David Schmidt, Simone Paolo Ponzetto, Goran Glavaš.


**Important notes**: This repository also contains the code for modified versions of two dependencies:

- a Knowledge Editing framework
  ([zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)), to include our
  BabelReFT proposed method and a new way to perform the evaluation (in [./EasyEdit](./EasyEdit/README.md))
- a page view counter for Wikipedia
  [gesiscss/wiki-donwload-parse-page-views](https://github.com/gesiscss/wiki-download-parse-page-views),
  to seperate couts per language (in [./wiki-count](./wiki-count/README.md))

## How to install

This code uses `uv` to deal with dependencies. In order to run the code with the
right dependencies, you simply need to run the following (or use any alternative
way to [install uv](https://docs.astral.sh/uv/getting-started/installation/))

```{bash}
pip install uv
```

Then, to run a script, instead of using the `python` executable, use `uv run`

## How to edit a model

If you want to edit a model with the BabelEdit dataset, you first need to download it:

```
uv run huggingface-cli download umanlp/babeledits --repo-type dataset  --local-dir datasets/
```

This will download the data directly to the `datasets` folder. You can select
another path if you want, but you'll need to change the default configuration in
`configs` accordingly, or use the `dataset=` option when running the executable.

To run BabelReFT editing on 100 sequential edits in English, use the following command:

```{bash}
PYTHONPATH="." uv run edit.py method="babelreft" max_edits=10 sequential=True return_edited_weights_at_end=True
```

Available options can be found by running `PYTHONPATH="." uv run edit.py --help` or simply looking into the `configs` folder.

`<DATASET_PATH>` should be the path to an editing dataset, a JSON file containing BabelEdits examples.

## How to run downstream evaluation

To run downstream evaluation, use the following command:

```{bash}
uv run exec_eval.py --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype='bfloat16' --batch_size 1 --device cuda:0 --tasks xquad_en --editing_dataset datasets/babeledits_test.json
```

This will evaluate the base model. `uv run exec_eval.py --help` shows more options. To evaluate an edited model, use the `--load_weights` option with the path to a model saved by `edit.py`, providing that `edit.py` was run with either `return_edited_weights=True` or `return_edited_weights_at_end=True sequential=True` (results will be stored in ./logs by default). Please note that `--model_args` is still needed since `edit.py` is only saving the modified parameters and not the whole model.

## Architecture

The repository contains the following important directories:

- `annotation`: contains the manual evaluation of our markup-based translation approach compared to vanilla translation
- `babeledits_pipeline`: contains the code for building the BabelEdits dataset, making it reproducible is a work in progress
- `configs`: contains YAML configs to parametrize `edit.py` using hydra
- `data_ppl`: contains data for evaluation perplexity before and after editing
- `data_splits`: contains code for generating data splits
- `easy_edit_adaptations`: contains code for plugging into EasyEdit (a dispatcher and some logging redirection)
- `EasyEdit`: contains our forked version of [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)
- `wiki-count`: contains our forked version of [gesiscss/wiki-donwload-parse-page-views](https://github.com/gesiscss/wiki-download-parse-page-views)

## Coming soon

- The full dataset hosted on HuggingFace
- A fully runnable pipeline for reproducing Babeledits construction or building variants
- mzrce integration

## Cite this repository

If you're using this repository, please cite the following work:

```
@inproceedings{green-etal-2025-babeledits,
    title = "{BABELEDITS}: A Benchmark and a Modular Approach for Robust Cross-lingual Knowledge Editing of Large Language Models",
    author = "Green, Tommaso  and
      Gaschi, F{\'e}lix  and
      Schmidt, Fabian David  and
      Ponzetto, Simone Paolo  and
      Glava{\v{s}}, Goran",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.438/",
    pages = "8342--8369",
    ISBN = "979-8-89176-256-5",
}
```