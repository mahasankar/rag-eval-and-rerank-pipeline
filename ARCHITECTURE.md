# System Architecture & Technical Design (`rag-eval-and-rerank-pipeline`)

This document outlines the architectural decisions and data flow implemented across the 5-day development lifecycle of the RAG pipeline.

## 1. Data Ingestion & Chunking Strategy
* **Extraction:** Standard PDF parsers struggle with code block font streams and formatting. We utilized **PyMuPDF (`fitz`)** to extract pristine raw text page-by-page.
* **Hierarchical Indexing:** Documents are parsed into parent nodes (`chunk_size=2048`) and child leaf nodes (`chunk_size=512`). During retrieval, leaf nodes provide high-precision vector matches while exposing broader parent context to the LLM.

## 2. Two-Tier Adaptive Hybrid Retrieval Pipeline
To solve the limitation of vector embeddings missing exact keyword matches (such as function names, variable types, or syntax symbols like `\0`), we implemented a hybrid retrieval strategy:
1. **Dense Retrieval:** Vector similarity search using HuggingFace BGE embeddings against ChromaDB.
2. **Sparse Retrieval:** Exact-match keyword retrieval using native `bm25s`.
3. **Rank Fusion:** Results from both spaces are combined using **Reciprocal Rank Fusion (RRF)** with a constant $k=60$.
4. **Cross-Encoder Reranking:** Candidate nodes are re-scored using `BAAI/bge-reranker-base` to filter out semantic noise.
5. **Adaptive Tier Expansion:** If the top reranker confidence score falls below `0.65` on the first pass, the pipeline automatically triggers a second pass with an expanded candidate pool ($k=12$).

## 3. Faithfulness & Negative Rejection Guardrails
To prevent hallucinations on out-of-scope or unanswerable queries:
* A strict **Confidence Threshold (`0.5`)** is enforced on reranked nodes.
* If top confidence scores drop below the threshold, generation is immediately aborted, returning a deterministic refusal message (*"I cannot find anything meaningful to retrieve from the dataset to answer this query safely."*).