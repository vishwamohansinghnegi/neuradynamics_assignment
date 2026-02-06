```
# Policy RAG Assistant (Prompt Engineering + RAG Mini Project)

This project is a small Retrieval-Augmented Generation (RAG) system that answers questions using company policy documents (Refund, Cancellation, Shipping, Payment).  
It retrieves relevant chunks from the documents and generates an answer grounded in that retrieved context.

Note: No policy documents were provided in the assignment, so I created a small set of realistic mock policy files in the `data/` folder to demonstrate the complete workflow.

---

## What this project does

- Loads policy documents from `data/` (`.md` / `.txt`)
- Splits documents into chunks (with overlap)
- Stores chunk embeddings in ChromaDB
- Retrieves top relevant chunks for a user question
- Uses a MiniLM cross-encoder reranker to improve retrieval quality
- Uses Gemini to generate answers using only the retrieved context
- Refuses when the documents do not contain the answer
- Runs through a simple CLI (no UI)

---

## Tech Stack

- LLM: Gemini (`langchain-google-genai`)
- Embeddings: MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- Vector DB: ChromaDB
- Reranking: CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Framework: LangChain

---

## Project Structure

```

NeuraDynamics_assignment/
│
├── data/                 # policy documents
├── chroma_db/            # generated after ingestion
├── eval/                 # evaluation set + script
├── src/                  # main code
│   ├── config.py
│   ├── ingest.py
│   ├── prompts.py
│   └── rag.py
│
├── app.py
├── requirements.txt
└── README.md

````

---

## Setup

### 1) Create and activate environment

```bash
python -m venv myenv
````

Windows:

```powershell
myenv\Scripts\activate
```

Mac/Linux:

```bash
source myenv/bin/activate
```

---

### 2) Install requirements

```bash
pip install -r requirements.txt
```

---

### 3) Add Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Run

### Step 1: Ingest documents into ChromaDB

```bash
python -m src.ingest
```

This creates the vector database in `chroma_db/`.

---

### Step 2: Start the CLI assistant

```bash
python app.py
```

Example questions:

* What is the refund window?
* Can I cancel after shipping?
* How long does delivery take?
* Is COD available?
* Do you support PayPal? (should refuse)

---

## How it works (simple)

1. Policy documents are loaded from `data/`
2. Text is chunked into smaller overlapping pieces
3. Chunks are embedded using MiniLM and stored in ChromaDB
4. When a question is asked:

   * Top-k chunks are retrieved using semantic search
   * Retrieved chunks are reranked using a cross-encoder
   * The final context is passed to Gemini
5. Gemini answers using a strict prompt to avoid hallucination

---

## Chunking Strategy

Configured in `src/config.py`

* Chunk size: 800
* Overlap: 150

Reason: policy documents contain short clauses. Smaller chunks improve retrieval accuracy, and overlap prevents missing important boundary sentences.

---

## Prompts Used

Stored in `src/prompts.py`

### Prompt v1 (Baseline)

* Basic “answer from context”
* Weak refusal handling

### Prompt v2 (Improved)

* Strict grounding rules (answer only from retrieved context)
* Explicit refusal when missing
* Structured output with citations

---

## Evaluation

Evaluation questions are in:

```
eval/questions.json
```

Run evaluation:

```bash
python eval/run_eval.py
```

The evaluation set includes:

* Answerable questions
* Partially answerable questions
* Unanswerable questions (to test hallucination avoidance)

---

## Key Trade-offs

* Kept the system small and readable (focus is retrieval + prompting).
* Added reranking for better relevance, but it increases runtime slightly.
* Used manual evaluation for transparency instead of automated scoring.

---

## Improvements with more time

* Add a query router:

  * normal greetings/general questions → normal assistant response
  * policy questions → strict RAG mode
* Add similarity thresholding to reduce irrelevant retrieval
* Add structured JSON output validation
* Add basic tracing/logging for retrieval and reranking scores

---

## What I’m most proud of

The improved prompt and grounding approach, which prevents hallucinations and forces safe refusals when the documents do not contain the answer.

---

## One thing I’d improve next

A small query router so that normal messages like “Hi” are handled naturally, while policy questions remain strictly grounded in the documents.

```
```
