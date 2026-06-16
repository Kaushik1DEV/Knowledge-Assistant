"""Retrieval eval harness. Run after any retrieval change to compare metrics:

    python eval/evaluate.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.rag import RAGEngine

GOLDEN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.csv")


def load_golden(path: str) -> list[dict]:
    df = pd.read_csv(path).replace({pd.NA: None, float("nan"): None})
    golden = df.to_dict(orient="records")
    for item in golden:
        v = item.get("expected_answer_contains")
        if isinstance(v, str):
            item["expected_answer_contains"] = ast.literal_eval(v)
    return golden


def evaluate_retrieval(engine: RAGEngine, golden: list[dict], k: int = 5):
    hits, rr = 0, []
    scored = [g for g in golden if g.get("expected_source")]
    for item in scored:
        retrieved = engine.retrieve_rerank(item["question"], k_final=k)
        sources = retrieved["source"].tolist()
        if item["expected_source"] in sources:
            hits += 1
            rr.append(1 / (sources.index(item["expected_source"]) + 1))
        else:
            rr.append(0)
    n = len(scored) or 1
    print(f"Hit-rate@{k}: {hits/n:.2f}   MRR: {sum(rr)/n:.2f}   (n={n})")


if __name__ == "__main__":
    evaluate_retrieval(RAGEngine(), load_golden(GOLDEN_CSV))
