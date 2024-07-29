# %%
import pandas as pd
from get_pages import process_multiple_pages
import os
lang = "de"
user_agent = os.getenv("WIKI_AGENT")

df = pd.read_csv(f"wikipedia_data/v5/raw/{lang}.csv", sep=" ", header=0, names=["Title", "Views"])
df = df.dropna().reset_index(drop=True)
df = df[pd.to_numeric(df['Views'], errors='coerce').notnull()]
df['Views'] = df['Views'].astype(int)
# %%
filtered_df = df[(df['Views'] >= 10000) & (df['Views'] <= 100000)]
filtered_df = filtered_df.sample(10000)
# %%
langs = ["en", f"{lang}"]
filtered_list = list(zip(filtered_df['Title'], filtered_df['Views']))
res = process_multiple_pages(filtered_list, lang, user_agent, langs)
res = res.dropna().sort_values("Views")
res.to_csv(f"hard_{lang}.csv", index=False, sep= " ")
# %%
from sentence_transformers import SentenceTransformer
from utils import clean
import pandas as pd
res =  pd.read_csv("wikipedia_data/v5/hard/hard_it.csv", sep= " ", header=0, names=["Title","Views","English Title","Languages"])
ent_1 = [clean(x) for x in res["Title"].values]
ent_2 = [clean(x) for x in res["English Title"].values]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings1 = model.encode(ent_1)
embeddings2 = model.encode(ent_2)
similarities = model.similarity_pairwise(embeddings1, embeddings2).numpy()
# %%
