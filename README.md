# Babeledits

This repository contains the code for building the BabelEdits dataset and for evaluating various Knowledge Editing methods on it. It relies on a modified version of [EasyEdit](https://github.com/zjunlp/EasyEdit), which the authors have no ties with, found in `./EasyEdit`.

This modified EasyEdit contains our code for the BabelReFT method, as well as some new evaluation metrics.

Finally this repository also contains code for evaluating edited models on downstream performance.

## How to install

This code uses `uv` to deal with dependencies. In order to run the code with the right dependencies, you simply need to run:

```{bash}
pip install uv
```

Then, to run a script, instead of using the `python` executable, use `uv run`

## How to edit a model

To run BabelReFT editing on 100 sequential edits in English, use the following command:

```{bash}
PYTHONPATH="." uv run edit.py dataset=<DATASET_PATH> method="babelreft" max_edits=100 sequential=True
```

Available options can be found by running `PYTHONPATH="." uv run edit.py --help` or simply looking into the `configs` folder.

`<DATASET_PATH>` should be the path to an editing dataset, a JSON file containing BabelEdits examples.

## How to run downstream evaluation

To run downstream evaluation, use the following command:

```{bash}
uv run exec_eval.py --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype='bfloat16' --batch_size 1 --device cuda:0 --tasks xquad_en --editing_dataset <DATASET_PATH>
```

This will evaluate the base model. `uv run exec_eval.py --help` shows more options. To evaluate an edited model, use the `--load_weights` option with the path to a model saved by `edit.py`, providing that `edit.py` was run with either `return_edited_weights=True` or `return_edited_weights_at_end=True sequential=True`. Please note that `--model_args` is still needed since `edit.py` is only saving the modified parameters and not the whole model.