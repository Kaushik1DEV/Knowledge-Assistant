"""Gradio UI for Hugging Face Spaces deployment (no Docker needed).

On HF Spaces: set AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT as Secrets,
commit data/knowledge.faiss + data/chunks.parquet, and this is the entrypoint.
"""
import gradio as gr

from app.rag import RAGEngine

rag = RAGEngine()  # loaded once when the Space boots


def ask(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."
    result = rag.answer(question)
    sources = ", ".join(sorted({c["source"] for c in result["citations"]}))
    grounded = "✅ grounded" if result["grounded"] else "⚠️ not answered from docs"
    return f"{result['answer']}\n\n---\n{grounded}  |  Sources: {sources or 'none'}"


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, label="Ask about leave, reimbursement, or insurance"),
    outputs=gr.Textbox(label="Answer"),
    title="HR Knowledge Assistant",
    description="RAG over leave, reimbursement & insurance policies. Cites sources.",
)

if __name__ == "__main__":
    demo.launch()
