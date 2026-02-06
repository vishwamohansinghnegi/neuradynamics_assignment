from dotenv import load_dotenv
from src.rag import answer


def main():
    load_dotenv()

    print("\nPolicy RAG Assistant (LangChain + Gemini + MiniLM + Reranking)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        a, retrieved = answer(q, prompt_version="improved")

        print("\nAssistant:\n" + a)

        print("\nRetrieved sources:")
        for i, d in enumerate(retrieved, start=1):
            print(f"  {i}. {d['source']}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
