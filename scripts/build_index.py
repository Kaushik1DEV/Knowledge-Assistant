"""OFFLINE indexing pipeline. Run whenever the source documents change:

    python scripts/build_index.py

Produces data/knowledge.faiss + data/chunks.parquet, which the API loads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import pandas as pd

from app.config import CHUNKS_PATH, DATA_DIR, INDEX_PATH
from app.embeddings import get_embedding
from app.ingest import build_chunks

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
DOCS = [
    os.path.join(DOCS_DIR, "FAQ_REIMBURSEMENT_POLICY.txt"),
    os.path.join(DOCS_DIR, "INSURANCE_POLICY.pdf"),
    os.path.join(DOCS_DIR, "LEAVE_POLICY.pdf"),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    chunks = build_chunks(DOCS)
    df = pd.DataFrame(chunks)
    print(f"Built {len(df)} chunks")

    embs = np.array([get_embedding(t) for t in df["text"]], dtype=np.float32)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    df.to_parquet(CHUNKS_PATH)  # note: embeddings NOT stored here (kept in FAISS)
    print(f"Saved index ({index.ntotal} vectors) -> {INDEX_PATH}")
    print(f"Saved chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
