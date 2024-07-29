from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils import clean
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Filter hard Wikipedia pages.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="wikipedia_data/v5/hard/processed",
        help="Path to the input data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="wikipedia_data/v5/hard/filtered",
        help="Path to save the output data",
    )
    parser.add_argument("--lang", type=str, help="Language code")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Cosine similarity threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda:idx)"
    )
    args = parser.parse_args()

    res = pd.read_csv(
        f"{args.data_path}/{args.lang}.csv",
        names=["Title", "Views", "English Title", "Languages"],
    )
    ent_1 = [clean(x) for x in res["Title"].values]
    ent_2 = [clean(x) for x in res["English Title"].values]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").to(args.device)

    embeddings1 = model.encode(ent_1)
    embeddings2 = model.encode(ent_2)
    res["cos_sim"] = model.similarity_pairwise(embeddings1, embeddings2).numpy()
    filtered = res[res["cos_sim"] <= args.threshold]
    print(f"Extracted {len(filtered)} pages for {args.lang}")
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    filtered.to_csv(f"{args.save_path}/{args.lang}.csv", index=False)


if __name__ == "__main__":
    main()
