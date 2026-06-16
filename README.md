# HR Knowledge Assistant (RAG)

A production-shaped RAG assistant over **leave, reimbursement & insurance** policies.
Q&A-aware chunking → FAISS retrieval → cross-encoder reranking → query
decomposition → groundedness + injection guardrails → cited answers.

## Project layout
```
knowledge-assistant/
├── app/
│   ├── config.py       # env-based config + Azure client (no hardcoded secrets)
│   ├── embeddings.py   # text -> vector
│   ├── ingest.py       # type-aware parsing + Q&A chunking
│   ├── rag.py          # RAGEngine: decompose -> retrieve -> rerank -> guard -> answer
│   ├── schemas.py      # Pydantic request/response
│   └── main.py         # FastAPI app (/health, /query)
├── scripts/build_index.py   # OFFLINE: docs -> FAISS + parquet
├── eval/evaluate.py         # retrieval metrics (Hit-rate, MRR)
├── app_gradio.py            # Hugging Face Spaces entrypoint
├── docs/                    # source policy files
├── data/                    # generated index (knowledge.faiss + chunks.parquet)
├── .env.example  .gitignore  requirements.txt
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env        # then paste your ROTATED Azure key into .env
```

## 1. Build the index (run once / when docs change)
```bash
python scripts/build_index.py
```

## 2a. Run the API locally
```bash
uvicorn app.main:app --reload
# open http://localhost:8000/docs  (interactive Swagger UI)
```

## 2b. Or run the UI
```bash
python app_gradio.py
```

## 3. Evaluate retrieval
Put your golden questions in `eval/golden.csv` with columns:
`question, expected_source, expected_answer_contains`
```bash
python eval/evaluate.py
```

## Deploy (no Docker) — Hugging Face Spaces
1. Create a **Gradio** Space.
2. Push this repo (include `data/knowledge.faiss` + `data/chunks.parquet`).
3. Set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` in the Space **Secrets**.
4. Space runs `app_gradio.py` → you get a public URL.

## Security notes
- Secrets only via `.env` / Space Secrets — never in code.
- Retrieved document text is treated as untrusted: delimited as data and
  screened for prompt-injection patterns before reaching the LLM.
- Answers are gated: out-of-context questions return "I don't know".
