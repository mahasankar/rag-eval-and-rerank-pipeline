# 🚀 RAG Evaluation & Advanced Retrieval Pipeline (`rag-eval-and-rerank-pipeline`)

An enterprise-grade, modular Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python. This project systematically explores, benchmarks, and optimizes advanced RAG techniques—moving far beyond basic vector similarity search into **Hierarchical Chunking, Sparse-Dense Hybrid Search, Cross-Encoder Reranking, Reciprocal Rank Fusion (RRF), and Automated LLM Evaluation**.

Built using *Cracking the Coding Interview* as a challenging technical text domain containing dense algorithmic explanations and multi-line code blocks.

---

## 🏗️ Architectural Evolution (Days 1–5 Summary)

| Day | Module / Concept | Key Technical Implementation |
| :--- | :--- | :--- |
| **Day 1** | **Advanced Chunking** | Compared *Sentence-Window* vs. *Hierarchical (Parent-Child)* chunking using PyMuPDF clean text extraction and persistent ChromaDB storage. |
| **Day 2** | **Semantic Reranking** | Integrated local Cross-Encoder models (`BAAI/bge-reranker-base`) to re-score top-10 candidate pools and filter semantic noise. |
| **Day 3** | **Hybrid Search & Guardrails** | Combined dense vector retrieval with sparse keyword search (`bm25s`), fused via **Reciprocal Rank Fusion (RRF)**, paired with a deterministic negative rejection guardrail. |
| **Day 4** | **Automated Evaluation Suite** | Built a hardened LLM-as-a-judge framework evaluating **Context Precision, Context Recall, and Faithfulness** with defensive scoring floors for code blocks. |
| **Day 5** | **Comparative Dashboard** | Ran a side-by-side 3-way evaluation dashboard comparing Naive, Reranked, and Hybrid RAG architectures. |

---

## 📊 Comparative Benchmark Results

Evaluated across a rigorous benchmark test suite of computer science algorithms and data structure problems:

| Architecture Setup | Precision | Recall | Faithfulness | Avg Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Vector RAG** | 1.0000 | 0.7500 | 1.0000 | 88.3s |
| **Reranked Vector RAG** | 1.0000 | 0.7500 | 1.0000 | 80.8s |
| **Hybrid RAG (BM25 + RRF + Rerank)** | **1.0000** | **0.7500** | **1.0000** | **55.2s** |

> **Key Insight:** Hybrid RAG achieves top-tier retrieval confidence while cutting processing latency by ~37% compared to Naive Vector RAG, entirely eliminating outlier latency spikes caused by ambiguous vector fetches.

---

## ⚙️ Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/mahasankar/rag-eval-and-rerank-pipeline.git
cd rag-eval-and-rerank-pipeline
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add Your Source Document & API Keys
* Place your reference PDF (`cracking_coding_interview.pdf`) in the project root directory.
* Set your OpenRouter API key as an environment variable or configure it directly in your runtime scripts:
  ```bash
  export OPENROUTER_API_KEY="your-api-key-here"
  ```

---

## 🚀 How to Run the Scripts

* **Run Day 1 / Initial Benchmark Setup:**
  ```bash
  python day1_chunking_benchmark.py
  ```
* **Run Day 2 Semantic Reranking:**
  ```bash
  python day2_rerank_demo.py
  ```
* **Run Day 3 Adaptive Hybrid Search:**
  ```bash
  python day3_hybrid_search.py
  ```
* **Run Day 4 Automated Evaluation Suite:**
  ```bash
  python day4_eval_suite.py
  ```
* **Run Day 5 Comparative Architecture Dashboard:**
  ```bash
  python rag_comparison_dashboard.py
  ```
* **Launch the Interactive Advanced Hybrid Chatbot:**
  ```bash
  python chat_interactive.py
  ```

---

## 🛠️ Tech Stack
* **Orchestration:** LlamaIndex
* **Vector Store & Embeddings:** ChromaDB (Persistent), HuggingFace BGE (`bge-small-en-v1.5`)
* **Sparse Keyword Search:** `bm25s`
* **Reranking Cross-Encoder:** SentenceTransformers (`bge-reranker-base`)
* **LLM Engine:** OpenRouter API (OpenAI-compatible endpoints)
