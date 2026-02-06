from dataclasses import dataclass

@dataclass
class Settings:
    data_dir: str = "data"
    chroma_dir: str = "chroma_db"
    collection_name: str = "company_policies"

    chunk_size: int = 800
    chunk_overlap: int = 150

    retrieve_top_k: int = 6
    rerank_top_k: int = 3

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Gemini LLM
    chat_model: str = "gemini-2.5-flash"

settings = Settings()