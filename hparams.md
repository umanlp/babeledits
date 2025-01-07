# Hparams

These hyperparameters were found using a wandb sweep gridsearch using the v8/translated/val_100.json dataset. Metric used is token_em_lm_eval.

## FT-L

### Llama
Best-found: method.layers=[21] method.lr=0.0005 method.norm_constraint=0.002
Score: 0.1554
Run name: fallen-sweep-16
Comment: By far the best, layer 31 seems to outperform all other layers. Top 3 results are in the layer range 29-31.
Updated: Layer 31 causes model collapse, switched to layer 21.

### Aya # TODO CHECK
Best-found: method.layers=[21] method.lr=0.0005 method.norm_constraint=0.002
Score: 0.1518
Run name: earthy-sweep-46
Comment: Much worse scorses than Llama. Layer 21 seems to be the best layer for Aya. No pattern for where is the best layer range.

### Gemma2 
Best-found: method.layers=[23] method.lr=0.0001 method.norm_constraint=0.002
Score: 0.1209
Run name: major-sweep-49
Comment: Middle region 19-30 seems to be the best. (gemma has 42 layers)


## FT-M

### Llama
Best-found: method.layers=[15] method.lr=0.0005
Score: 0.4618
Run name: astral-sweep-10
Comment: Best layers seem mid-to-late. Much higher scores than other layers. Again, top layers seem to be the best.
Update: Layer 31 causes model collapse, switched to layer 15.

### Aya # TODO CHECK
Best-found: method.layers=[31] method.lr=0.0005
Score: 0.4018
Run name: cerulean-sweep-26
Comment: Best layers seem mid-to-late. Same combination as Llama, but with lower scores.

### Gemma2 
Best-found: method.layers=[27] method.lr=0.0005
Score: 0.4463
Run name: lyric-sweep-22
Comment: Middle region 23-30 seems to be the best. (gemma has 42 layers)


## R-ROME

### Llama

Best-found: method.kl_factor=0.0625 method.layers=[15]
Score: 0.3727
Run name: pleasant-sweep-5
Comment: Low-to-mid layers are the best. Layer 15 seems to be the best layer.

### Aya

Best-found: method.kl_factor=0.0625 method.layers=[7]
Score: 0.267
Run name: classic-sweep-4
Comment: Early layers are the best, around 5-7.

### Gemma2 
Best-found: method.kl_factor=0.9 method.layers=[25]
Score: 0.2236
Run name: stellar-sweep-23
Comment: Rather weak scores, but the best layers seem to be in the middle region 20-27. (gemma has 42 layers)

## Babelreft (all layers all tokens)

### Llama 
Score: 0.416
Run name: blooming-sweep-3
Best-found: method.low_rank_dim=4 method.lr=0.002 

### Gemma2
Score: 0.417
Run name: worldly-sweep-7
Best-found: method.low_rank_dim=16 method.lr=0.001

## Summary Table # TODO UPDATE

| Method | Model | Best-found Hyperparameters | Score  | Run name         | Comment                                                                 |
|--------|-------|----------------------------|--------|------------------|-------------------------------------------------------------------------|
| FT-L   | Llama | layers=[31], lr=0.0005, norm_constraint=0.002 | 0.2518 | deft-sweep-46    | By far the best, layer 31 seems to outperform all other layers. Top 3 results are in the layer range 29-31. |
| FT-L   | Aya   | layers=[21], lr=0.0005, norm_constraint=0.002 | 0.1518 | earthy-sweep-46  | Much worse scores than Llama. Layer 21 seems to be the best layer for Aya. No pattern for where is the best layer range. |
| FT-M   | Llama | layers=[31], lr=0.0005     | 0.47   | fancy-sweep-16   | Best layers seem mid-to-late. Much higher scores than other layers. Again, top layers seem to be the best. |
| FT-M   | Aya   | layers=[31], lr=0.0005     | 0.4018 | cerulean-sweep-26| Best layers seem mid-to-late. Same combination as Llama, but with lower scores. |
| R-ROME | Llama | kl_factor=0.0625, layers=[15] | 0.3727 | pleasant-sweep-5 | Low-to-mid layers are the best. Layer 15 seems to be the best layer. |
| R-ROME | Aya   | kl_factor=0.0625, layers=[7]  | 0.267  | classic-sweep-4  | Early layers are the best, around 5-7. |