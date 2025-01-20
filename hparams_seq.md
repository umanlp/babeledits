# Hparams

These hyperparameters were found using a wandb sweep gridsearch using the v8/translated/val_100.json dataset. Metric used is token_em_lm_eval.

## FT-L

### Llama
Best-found: method.layers=[19] method.lr=0.0001 method.norm_constraint=0.002 sequential=true
Rewrite Score: 0.1554
Run name: cosmic-sweep-37
Comment: 

### Gemma2 


## FT-M

### Llama
Best-found: method.layers=[21] method.lr=0.0005 model=llama-3-1 method=ft sequential=true
Rewrite Score: 0.278
Run name: gentle-sweep-16
Comment: 


### Gemma2 


## R-ROME

### Llama

Best-found: method.kl_factor=1 method.layers=[17] model=llama-3-1 method=r-rome sequential=true
Score: 0.009
Run name: polar-sweep-32
Comment: 

### Gemma2 

## Babelreft (all layers all tokens)

### Llama 
Score: 0.416
Run name: blooming-sweep-3
Best-found: method.low_rank_dim=4 method.lr=0.002 

### Gemma2

