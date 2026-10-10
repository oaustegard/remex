"""RemexIndex recall@10 against float32 util.semantic_search.

10,000 Natural Questions answers as the corpus, 500 held-out questions as
queries, all-MiniLM-L6-v2. Needs sentence-transformers, pandas and network
access to the Hugging Face hub.

    python bench/st_eval.py
"""
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util

from remex import RemexIndex


def main(n_corpus=10_000, n_queries=500, model_name="all-MiniLM-L6-v2"):
    path = hf_hub_download(
        "sentence-transformers/natural-questions",
        "pair/train-00000-of-00001.parquet", repo_type="dataset",
    )
    df = pd.read_parquet(path)
    docs = df["answer"].tolist()[:n_corpus]
    queries = df["query"].tolist()[n_corpus:n_corpus + n_queries]

    model = SentenceTransformer(model_name)
    D = model.encode(docs, batch_size=128, normalize_embeddings=True)
    Q = model.encode(queries, normalize_embeddings=True)
    truth = util.semantic_search(torch.tensor(Q), torch.tensor(D), top_k=10)
    truth = [{h["corpus_id"] for h in hits} for hits in truth]

    print(f"{model_name}, {len(docs)} docs, {len(queries)} queries")
    print("bits  center  R@10   ratio")
    for bits in (1, 2, 3, 4, 8):
        for center in (False, True):
            index = RemexIndex(model, bits=bits, center=center)
            index.add(D)
            got = index.search(Q, top_k=10)
            recall = np.mean([
                len(t & {h["corpus_id"] for h in g}) / 10
                for t, g in zip(truth, got)
            ])
            print(f"{bits:>4}  {center!s:>6}  {recall:.3f}  "
                  f"{index.compression_ratio:.1f}x")


if __name__ == "__main__":
    main()
