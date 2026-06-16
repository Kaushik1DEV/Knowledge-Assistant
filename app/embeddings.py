"""Embedding helper — single source of truth for turning text into vectors."""
from .config import get_client, EMBED_MODEL


def get_embedding(text: str) -> list[float]:
    resp = get_client().embeddings.create(input=text, model=EMBED_MODEL)
    return resp.data[0].embedding
