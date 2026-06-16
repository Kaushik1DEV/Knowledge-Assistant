"""FastAPI serving layer. Loads the RAG engine ONCE at startup."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .rag import RAGEngine
from .schemas import QueryRequest, QueryResponse

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RAGEngine()  # runs once: load index + reranker into memory
    yield
    state.clear()


app = FastAPI(title="Knowledge Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": int(state["rag"].df.shape[0])}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    try:
        return state["rag"].answer(req.question)
    except Exception as e:  # never leak stack traces to clients
        raise HTTPException(500, "Error processing query") from e
