"""
rag_engine.py — Retrieval Augmented Generation (RAG) Engine
============================================================
WHAT THIS FILE DOES:
    Answers plain-English questions about HR documents (policies, benefits,
    code of conduct) by retrieving the most relevant text chunks from
    ChromaDB and passing them to GPT-4o as context.

THE PIPELINE (end to end):
    User question
         │
         ▼
    [1] Embed the question → 384-dim vector (same model used during ingestion)
         │
         ▼
    [2] ChromaDB cosine similarity search → top-k most relevant chunks
         │
         ▼
    [3] Build a prompt that "stuffs" those chunks in as context
         │
         ▼
    [4] GPT-4o generates an answer grounded ONLY in the retrieved chunks
         │
         ▼
    [5] Return structured result: answer + source doc names + chunks used

WHY "GROUNDED ONLY IN RETRIEVED CHUNKS"?
    We explicitly tell the LLM: "If the answer is not in the documents,
    say you don't know." This is the key defence against hallucination.
    The LLM must cite what it was given, not invent an answer.

JAVA ANALOGY:
    This is a Spring @Service that:
      1. Calls a SearchRepository (ChromaDB) to load relevant documents
      2. Builds a request DTO (prompt) from those documents
      3. Calls an external REST API (OpenAI GPT-4o)
      4. Returns a structured response DTO

RETURN VALUE SHAPE (dict):
    {
        "answer":      str   — human-readable answer from the LLM,
        "sources":     list  — unique document filenames used,
        "chunks_used": list  — raw text of each chunk passed to the LLM,
        "error":       str   — error message if something failed, else None
    }
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
import traceback
from typing import Optional

# ── LangChain: prompt template ───────────────────────────────────────────────
# PromptTemplate is LangChain's way of building parameterised prompts.
# You define a template string with {placeholders}, and fill them at runtime.
# Java analogy: like String.format() but with named keys, not positional %s
from langchain_core.prompts import PromptTemplate

# ── LangChain: output parser ─────────────────────────────────────────────────
# StrOutputParser extracts the plain text content from an LLM response object.
# Without it, the chain returns an AIMessage object; with it, you get a string.
from langchain_core.output_parsers import StrOutputParser

# ── Our own modules ──────────────────────────────────────────────────────────
from app.engines.llm import get_llm
from app.vectorstore.store import retrieve_with_scores


# ── Constants ────────────────────────────────────────────────────────────────

# How many chunks to retrieve from ChromaDB.
# More chunks = more context but more tokens consumed and possible noise.
# 5 is a good default for HR policy docs (each chunk is ~800 chars).
_DEFAULT_K = 5

# Minimum similarity score to accept a chunk (0.0 = accept all, 1.0 = exact match).
# ChromaDB returns cosine distances (lower = more similar).
# We filter out anything with distance > 0.7 (poor relevance).
_MAX_DISTANCE = 0.7

# ── The RAG prompt template ──────────────────────────────────────────────────
# This is the "stuffing" prompt — we paste retrieved chunks directly into it.
#
# KEY DESIGN CHOICES in this prompt:
#  1. "Use ONLY the documents provided" → prevents the LLM from mixing in
#     its own training knowledge (hallucination prevention)
#  2. "If the answer is not in the documents, say: I don't have that information"
#     → graceful failure instead of a confident wrong answer
#  3. We list sources so the LLM (and user) can see where the answer came from
#
_RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful HR assistant for a company. Answer the employee's
question using ONLY the information provided in the documents below.

If the answer is not found in the documents, respond with:
"I don't have that information in the available HR documents."

Do NOT use any knowledge outside of the provided documents.
Be concise and factual. If quoting a policy, state it clearly.

--- DOCUMENTS ---
{context}
--- END OF DOCUMENTS ---

Employee question: {question}

Answer:""",
)


# ── Helper: format retrieved chunks into the {context} string ────────────────

def _format_context(chunks_with_scores: list[tuple]) -> tuple[str, list[str], list[str]]:
    """
    Takes the raw output of retrieve_with_scores() and returns three things:
      - context_str : the formatted text block to inject into the prompt
      - sources     : unique list of source document filenames
      - chunk_texts : raw text of each chunk (for the return dict)

    retrieve_with_scores() returns a list of (Document, distance) tuples.
    A LangChain Document has:
      - page_content : str  — the chunk text
      - metadata     : dict — contains "source" key with the filename
    """
    context_parts: list[str] = []
    sources: list[str] = []
    chunk_texts: list[str] = []

    for doc, distance in chunks_with_scores:
        # Skip chunks that are too dissimilar (low relevance)
        if distance > _MAX_DISTANCE:
            continue

        text = doc.page_content.strip()
        # metadata["source"] is the full file path — we only want the filename
        # Example: "data/documents/hr_leave_policy.txt" → "hr_leave_policy.txt"
        source = doc.metadata.get("source", "unknown")
        filename = source.split("/")[-1].split("\\")[-1]  # works on both / and \

        # Format each chunk with its source label so the LLM knows where it came from
        context_parts.append(f"[Source: {filename}]\n{text}")

        chunk_texts.append(text)

        # Collect unique source names (don't add duplicates)
        if filename not in sources:
            sources.append(filename)

    context_str = "\n\n".join(context_parts)
    return context_str, sources, chunk_texts


# ── Main public function ─────────────────────────────────────────────────────

def run_rag_query(question: str, k: int = _DEFAULT_K) -> dict:
    """
    Full RAG pipeline: question → ChromaDB retrieval → LLM → structured answer.

    Parameters
    ----------
    question : str
        The plain-English question from the user.
    k : int
        Number of top chunks to retrieve. Default: 5.

    Returns
    -------
    dict with keys:
        answer      (str)  — LLM-generated answer grounded in documents
        sources     (list) — unique document filenames used
        chunks_used (list) — raw text of each chunk passed to the LLM
        error       (str | None) — error message if failed, else None
    """
    try:
        # ── STEP 1: Retrieve relevant chunks from ChromaDB ───────────────────
        # retrieve_with_scores() returns [(Document, cosine_distance), ...]
        # Lower distance = more similar. Range: 0.0 (identical) to 2.0 (opposite).
        chunks_with_scores = retrieve_with_scores(question, k=k)

        if not chunks_with_scores:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "chunks_used": [],
                "error": None,
            }

        # ── STEP 2: Format chunks into the {context} block ───────────────────
        context_str, sources, chunk_texts = _format_context(chunks_with_scores)

        if not context_str.strip():
            # All chunks were filtered out by the distance threshold
            return {
                "answer": "I don't have sufficiently relevant information in the HR documents to answer that question.",
                "sources": [],
                "chunks_used": [],
                "error": None,
            }

        # ── STEP 3: Build the chain and invoke it ────────────────────────────
        # Chain = prompt template | LLM | output parser
        #
        # The pipe | operator is LCEL (LangChain Expression Language).
        # It's equivalent to: parser.parse(llm.call(prompt.format(context, question)))
        # But written left-to-right as a readable pipeline.
        #
        # Java analogy:
        #   _RAG_PROMPT_TEMPLATE  →  builds the request (like constructing a DTO)
        #   get_llm(temperature=0.3)  →  calls the external API
        #   StrOutputParser()  →  extracts the response body as a plain string
        llm = get_llm(temperature=0.3)
        chain = _RAG_PROMPT_TEMPLATE | llm | StrOutputParser()

        # chain.invoke() executes all three steps in sequence.
        # We pass a dict matching the {placeholders} in the prompt template.
        answer = chain.invoke({
            "context": context_str,
            "question": question,
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "chunks_used": chunk_texts,
            "error": None,
        }

    except Exception as exc:
        # Catch-all: log the full traceback for debugging, return structured error
        traceback.print_exc()
        return {
            "answer": None,
            "sources": [],
            "chunks_used": [],
            "error": str(exc),
        }
