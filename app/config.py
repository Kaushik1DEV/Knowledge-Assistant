"""Central configuration. Secrets come from .env — never hardcoded."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # reads .env in project root

# --- Secrets (required) ---
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

# --- Tunables (sensible defaults) ---
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "r1chartmanager-hl7parser-40")

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.parquet")

K_WIDE = int(os.getenv("K_WIDE", "20"))    # FAISS wide net
K_FINAL = int(os.getenv("K_FINAL", "5"))   # kept after reranking


@lru_cache  # one client, reused everywhere
def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
    )
