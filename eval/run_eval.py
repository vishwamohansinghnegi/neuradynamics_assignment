import json
import os
from pathlib import Path
from dotenv import load_dotenv

from src.rag import answer


def main():
    load_dotenv()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("Missing GOOGLE_API_KEY (or GEMINI_API_KEY).")
        return

    questions_path = Path(__file__).parent / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for i, item in enumerate(questions, start=1):
        q = item["question"]
        expected = item["expected"]

        ans, _ = answer(q, prompt_version="improved")

        print(f"\nQ{i}: {q}")
        print(f"Expected: {expected}")
        print(f"Answer: {ans}")


if __name__ == "__main__":
    main()
