"""
Advanced Interactive RAG Chatbot (Powered by Adaptive Hybrid Search & Cross-Encoders)
- Replaces basic vector chat with Day 3's Hybrid Search + RRF + Reranking.
- Includes Faithfulness Guardrail to reject out-of-scope queries safely.
"""

import os
import time
import bm25s
import chromadb

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

# Import the hardened adaptive hybrid search pipeline from Day 3
from day3_hybrid_search import adaptive_hybrid_search

def main():
    print("Initializing local embedding model (HuggingFace BGE)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Connecting to OpenRouter free-tier engine...")
    llm = OpenAILike(
        model="nvidia/nemotron-3.5-lightning:free",  
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        is_chat_model=True
    )

    # print("Connecting to Gemini LLM engine...")
    # llm = GoogleGenAI(
    #             model="gemini-2.5-flash",
    #             api_key=os.environ.get("GOOGLE_API_KEY")
    #         )
    
    # print("Connecting to Groq LLM engine (openai/gpt-oss-20b)...")
    # llm = Groq(
    #     model="openai/gpt-oss-20b",
    #     api_key=os.environ.get("GROQ_API_KEY")
    # )
    
    Settings.llm = llm

    print("Connecting to Persistent Hierarchical Vector Store...")
    db = chromadb.PersistentClient(path="./chroma_db_hierarchical")
    try:
        chroma_collection = db.get_collection("hierarchical_full")
    except Exception:
        print("\n[!] Error: Found no existing hierarchical database. Please run Day 1 setup first!")
        return

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print("Extracting nodes for native bm25s keyword engine...")
    all_nodes_result = index.as_retriever(similarity_top_k=chroma_collection.count()).retrieve(" ")
    index_nodes = [node.node for node in all_nodes_result]
    node_texts = [node.get_content() for node in index_nodes]

    print("Building BM25 keyword index...")
    tokenized_corpus = bm25s.tokenize(node_texts, stopwords="english")
    bm25_model = bm25s.BM25()
    bm25_model.index(tokenized_corpus)

    print("\n" + "="*60)
    print("💡 Advanced Hybrid RAG Chatbot (Cracking the Coding Interview)")
    print("Pipeline: Dense Vectors + Sparse BM25 + RRF + Cross-Encoder Rerank")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            query = input("\nUser Query: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("\nExiting chat. Good luck with your interview prep!")
                break
            if not query:
                continue

            print("\nSearching & Reranking across hybrid corpus...")
            start_time = time.time()

            # Execute Adaptive Hybrid Search
            reranked_nodes, final_top_score = adaptive_hybrid_search(query, index, index_nodes, bm25_model)

            # Faithfulness Guardrail Threshold
            GUARDRAIL_THRESHOLD = 0.5
            if final_top_score < GUARDRAIL_THRESHOLD:
                response = "I cannot find anything meaningful to retrieve from the dataset to answer this query safely."
                retrieved_contexts = []
            else:
                retrieved_contexts = [n.node.get_content() for n in reranked_nodes]
                context_str = "\n\n".join(retrieved_contexts)
                prompt = f"""Based on the following context from Cracking the Coding Interview, answer the query accurately with code implementations where applicable.

Context: 
{context_str}

Query: {query}
"""
                response = Settings.llm.complete(prompt)

            elapsed = time.time() - start_time
            print(f"\n[Hybrid RAG Response] (Top Rerank Score: {final_top_score:.2f} | Time: {elapsed:.2f}s):\n")
            print(response)

            print("\n--- [Top Retrieved Context References] ---")
            for i, node in enumerate(reranked_nodes):
                score = node.score if hasattr(node, 'score') and node.score is not None else 0.0
                #print(f"Ref {i+1} (Score: {score:.4f}): {node.node.get_content()[:180]}...")
                print(f"Ref {i+1} (Score: {score:.4f}): {node.node.get_content()}...")
            print("-" * 45)

        except KeyboardInterrupt:
            print("\n\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    main()