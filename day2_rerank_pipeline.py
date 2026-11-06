"""
Day 2: Semantic Reranking with Local Cross-Encoders
- Loads existing hierarchical ChromaDB index.
- Retrieves top 10 candidate nodes via vector similarity.
- Applies BGE Cross-Encoder Reranker to re-score and filter down to top 3.
- Synthesizes answer using Gemini.
"""

import os
import time
import chromadb

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.postprocessor import SentenceTransformerRerank

from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.groq import Groq

def main():
    print("Initializing local embedding model (HuggingFace BGE)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Connecting to OpenRouter free-tier engine...")
    llm = OpenAILike(
        model="nvidia/nemotron-3.5-lightning:free",  
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        is_chat_model=True  # Tells LlamaIndex to use OpenRouter's chat completions endpoint
        )

    #print("Connecting to Gemini LLM engine...")
    # llm = GoogleGenAI(
    #       model="gemini-2.5-flash",
    #       api_key=os.environ.get("GOOGLE_API_KEY")
    #       )

    # print("Connecting to Groq LLM engine (openai/gpt-oss-20b)...")
    # llm = Groq(
    #     model="openai/gpt-oss-20b",
    #     api_key=os.environ.get("GROQ_API_KEY")  
    # )

    Settings.llm = llm

    print("\nInitializing Local Cross-Encoder Reranker (BAAI/bge-reranker-base)...")
    # This lightweight reranker runs locally on CPU/GPU and scores query-node pairs with high precision
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=3  # Keep only the top 3 most relevant chunks after reranking
    )

    print("Connecting to Persistent Hierarchical Vector Store...")
    db = chromadb.PersistentClient(path="./chroma_db_hierarchical")
    chroma_collection = db.get_collection("hierarchical_full")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # Configure query engine with a wider initial pool (similarity_top_k=10) 
    # and attach our reranker postprocessor to filter down to the best 3.
    query_engine = index.as_query_engine(
        similarity_top_k=10,  # Cast a wide net first
        node_postprocessors=[reranker]  # Rerank and filter down to top_n=3
    )

    test_query = "What are the key tips for tackling technical interview questions?"
    print(f"\nQuery: '{test_query}'")
    
    start_time = time.time()
    response = query_engine.query(test_query)
    print(f"[Timing] Reranked query executed in {time.time() - start_time:.2f} seconds.")

    print("\n----------------------------------------------")
    print("[RETRIEVAL INSPECTION WITH CROSS-ENCODER RERANKING]:")
    print("----------------------------------------------")
    for i, node in enumerate(response.source_nodes):
        # Display the new rerank score provided by the cross-encoder
        score = node.score if hasattr(node, 'score') and node.score is not None else 0.0
        print(f"\n[Reranked Chunk {i+1} | Score: {score:.4f}]")
        #print(node.node.get_content()[:350] + "...")
        print(node.node.get_content() + "...")

    print("\n----------------------------------------------")
    print("[Final Generated LLM Response (Post-Rerank)]:")
    print("----------------------------------------------")
    print(response)

if __name__ == "__main__":
    main()