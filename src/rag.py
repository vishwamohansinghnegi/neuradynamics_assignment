from typing import List, Dict, Tuple

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.prompts import BASELINE_PROMPT, IMPROVED_PROMPT


# ----------------------------
# Load once (fast + standard)
# ----------------------------
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

vectordb = Chroma(
    persist_directory=settings.chroma_dir,
    embedding_function=embeddings,
    collection_name=settings.collection_name
)

retriever = vectordb.as_retriever(search_kwargs={"k": settings.retrieve_top_k})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatGoogleGenerativeAI(model=settings.chat_model, temperature=0)

parser = StrOutputParser()


# ----------------------------
# Small helpers (simple + reusable)
# ----------------------------
def rerank(question: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []

    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[: settings.rerank_top_k]]


def build_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] {src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ----------------------------
# Main public function
# ----------------------------
def answer(query: str, prompt_version: str = "improved") -> Tuple[str, List[Dict]]:
    # 1) Retrieve
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant policy information found for this question.", []

    # 2) Rerank
    docs = rerank(query, docs)

    if not docs:
        return "No relevant policy information found for this question.", []

    # 3) Build prompt
    context = build_context(docs)

    if prompt_version == "baseline":
        prompt = BASELINE_PROMPT.format(context=context, question=query)
    else:
        prompt = IMPROVED_PROMPT.format(context=context, question=query)

    # 4) Runnable chain (simple)
    chain = RunnableLambda(lambda x: x) | llm | parser
    response = chain.invoke(prompt).strip()

    # return docs for printing
    retrieved = [{"text": d.page_content, "source": d.metadata.get("source", "unknown")} for d in docs]
    return response, retrieved
