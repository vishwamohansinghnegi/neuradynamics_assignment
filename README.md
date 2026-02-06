## Policy RAG Assistant

### (Prompt Engineering + RAG Mini Project)

This project is a compact **Retrieval-Augmented Generation (RAG)** system designed to answer questions using company policy documents (Refund, Cancellation, Shipping, Payment). It retrieves relevant segments from the documentation and generates answers strictly grounded in the retrieved context.

> **Note:** As no policy documents were provided, a set of realistic mock policy files has been included in the `data/` folder to demonstrate the full workflow.

---

### ## What this project does

* **Data Loading:** Ingests policy documents from `data/` (`.md` / `.txt`).
* **Vector Storage:** Splits documents into overlapping chunks and stores embeddings in **ChromaDB**.
* **Advanced Retrieval:** Employs a **MiniLM cross-encoder reranker** to prioritize the most relevant chunks.
* **Grounded Generation:** Uses **Gemini** to generate answers using *only* the provided context.
* **Hallucination Guardrails:** Explicitly refuses to answer if the documents do not contain the necessary information.
* **Interface:** Operates via a streamlined Command Line Interface (CLI).

---

### ## Tech Stack

* **LLM:** Gemini (`langchain-google-genai`)
* **Embeddings:** MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector DB:** ChromaDB
* **Reranking:** CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **Framework:** LangChain

---

### ## Project Structure

```text
NeuraDynamics_assignment/
│
├── data/                 # Policy documents
├── chroma_db/            # Generated after ingestion
├── eval/                 # Evaluation set + script
├── src/                  # Main source code
│   ├── config.py
│   ├── ingest.py
│   ├── prompts.py
│   └── rag.py
│
├── app.py
├── requirements.txt
└── README.md

```

---

### ## Setup & Installation

#### 1. Create and activate environment

```bash
python -m venv myenv
# Windows:
myenv\Scripts\activate
# Mac/Linux:
source myenv/bin/activate

```

#### 2. Install requirements

```bash
pip install -r requirements.txt

```

#### 3. Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here

```

---

### ## Usage

**Step 1: Ingest documents** Process the raw text into the vector database:

```bash
python -m src.ingest

```

**Step 2: Start the Assistant** Launch the CLI to ask questions:

```bash
python app.py

```

**Example Queries:**

* "What is the refund window?"
* "Can I cancel after shipping?"
* "Do you support PayPal?" *(System should refuse if not in docs)*

---

### ## Key Features & Logic

* **Chunking Strategy:** Configured with a size of **800** and an overlap of **150**. This ensures that short policy clauses are captured entirely without losing context at the boundaries.
* **Dual-Stage Retrieval:** Semantic search identifies candidates; the Cross-Encoder ensures the LLM only sees the most statistically relevant information.
* **Prompt Engineering:** The system uses "Prompt v2," which includes strict grounding rules and structured output requirements to ensure safety and reliability.

---

### ## Evaluation & Future Improvements

The system is tested against `eval/questions.json`, covering answerable, partially answerable, and unanswerable cases.

**Future Roadmap:**

* **Query Routing:** Distinguish between "Hello" and "What is the shipping policy?" to handle general chat naturally.
* **Similarity Thresholds:** Automatically reject low-confidence retrievals before they reach the LLM.
* **Tracing:** Implement logging for reranking scores to audit retrieval quality.

---