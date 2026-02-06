from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def ingest():
    load_dotenv()

    files = list(Path(settings.data_dir).glob("*.md")) + list(Path(settings.data_dir).glob("*.txt"))
    if not files:
        raise RuntimeError(f"No documents found in {settings.data_dir}/")

    docs = []
    for f in sorted(files):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": f.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name
    )
    vectordb.persist()

    print("Ingestion complete")
    print(f"Files loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chroma saved at: {settings.chroma_dir}/")


if __name__ == "__main__":
    ingest()
