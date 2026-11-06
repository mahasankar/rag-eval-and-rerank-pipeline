"""
Day 3: Adaptive Dense + Sparse Hybrid Search with Negative Rejection Guardrail
- Uses native `bm25s` for fast keyword retrieval and LlamaIndex Vector Retriever.
- Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranking.
- Adaptive two-tier pool expansion with best-score tracking across attempts.
- Deterministic negative rejection fallback to eliminate hallucinations (Threshold = 0.5).
"""

import os
import time
import bm25s
import chromadb

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

def adaptive_hybrid_search(query, index, index_nodes, bm25_model):
    """
    Executes the two-tier adaptive hybrid search (Dense + BM25 + RRF + Rerank)
    with best-score tracking across attempts.
    """
    best_reranked_nodes = []
    best_top_score = -1.0
    
    current_top_n = 3
    initial_pool_k = 5

    for attempt in range(2):
        print(f"\n[Adaptive Pass {attempt+1}] Retrieving with pool_k={initial_pool_k}, top_n={current_top_n}...")
        
        # 1. Dense Vector Retrieval (Semantic)
        vector_retriever = index.as_retriever(similarity_top_k=initial_pool_k)
        dense_nodes = vector_retriever.retrieve(query)

        # 2. Sparse BM25 Retrieval (Keyword / Code snippets)
        query_tokens = bm25s.tokenize(query)
        results, scores = bm25_model.retrieve(query_tokens, k=initial_pool_k)
        sparse_nodes = [index_nodes[idx] for idx in results[0]]

        # 3. Combine via Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        fused_scores = {}
        node_map = {}

        def add_to_fusion(nodes):
            for rank, node in enumerate(nodes):
                node_id = node.node.node_id if hasattr(node, 'node') else node.node_id
                actual_node = node.node if hasattr(node, 'node') else node
                node_map[node_id] = actual_node
                score = 1.0 / (rrf_k + rank + 1)
                fused_scores[node_id] = fused_scores.get(node_id, 0.0) + score

        add_to_fusion(dense_nodes)
        add_to_fusion(sparse_nodes)

        sorted_node_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        
        combined_node_with_scores = [
            NodeWithScore(node=node_map[nid], score=fused_scores[nid]) 
            for nid in sorted_node_ids[:(initial_pool_k + 3)]
        ]

        # 4. Apply Cross-Encoder Reranking
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=current_top_n
        )
        current_reranked = reranker.postprocess_nodes(combined_node_with_scores, query_str=query)

        # 5. Intrinsic Confidence Check
        current_top_score = current_reranked[0].score if current_reranked and hasattr(current_reranked[0], 'score') and current_reranked[0].score is not None else 0.0
        print(f"  -> Top Reranker Confidence Score: {current_top_score:.4f}")

        # Track the absolute best result across attempts (never downgrade)
        if current_top_score > best_top_score:
            best_top_score = current_top_score
            best_reranked_nodes = current_reranked

        # Adaptive trigger threshold (0.65)
        if current_top_score < 0.65 and attempt == 0:
            print("  [Adaptive Trigger] Low confidence detected! Expanding search pool and window for complex multi-page context...")
            current_top_n = 6
            initial_pool_k = 12
            continue
        break 

    return best_reranked_nodes, best_top_score

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
    #         model="gemini-2.5-flash",
    #         api_key=os.environ.get("GOOGLE_API_KEY")
    #     )

    # print("Connecting to Groq LLM engine (openai/gpt-oss-20b)...")
    # llm = Groq(
    #     model="openai/gpt-oss-20b",
    #     api_key=os.environ.get("GROQ_API_KEY")
    # )

    Settings.llm = llm

    print("\nConnecting to Persistent Hierarchical Vector Store...")
    db = chromadb.PersistentClient(path="./chroma_db_hierarchical")
    chroma_collection = db.get_collection("hierarchical_full")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print("Extracting nodes for native bm25s keyword engine...")
    all_nodes_result = index.as_retriever(similarity_top_k=chroma_collection.count()).retrieve(" ")
    index_nodes = [node.node for node in all_nodes_result]
    node_texts = [node.get_content() for node in index_nodes]

    # Initialize and index corpus with bm25s
    print("Building BM25 keyword index...")
    tokenized_corpus = bm25s.tokenize(node_texts, stopwords="english")
    bm25_model = bm25s.BM25()
    bm25_model.index(tokenized_corpus)

    # Test query parameter
    test_query = "Describe an algorithm to set row and column to 0 if an element in an MxN matrix is 0"
    print(f"\nAdaptive Hybrid Search Query: '{test_query}'")
    
    start_time = time.time()

    # Call the modularized search function with best-score tracking
    reranked_nodes, final_top_score = adaptive_hybrid_search(test_query, index, index_nodes, bm25_model)

    # --- FAITHFULNESS NEGATIVE REJECTION GUARDRAIL ---
    CONFIDENCE_THRESHOLD = 0.5  

    if final_top_score < CONFIDENCE_THRESHOLD:
        print(f"\n[Guardrail Triggered] Final confidence score ({final_top_score:.4f}) is below acceptable threshold ({CONFIDENCE_THRESHOLD}). Aborting generation to prevent hallucination.")
        response = "I cannot find anything meaningful to retrieve from the dataset to answer this query safely."
        reranked_nodes = []
    else:
        # Synthesize final answer using LLM
        context_str = "\n\n".join([n.node.get_content() for n in reranked_nodes])
        

        prompt = f"""Based on the following context from Cracking the Coding Interview, answer the query accurately with code implementations where applicable.

                    Context: {context_str} Query: {test_query}
                  """
        response = Settings.llm.complete(prompt)
    
    print(f"\n[Timing] Adaptive hybrid query executed in {time.time() - start_time:.2f} seconds.")

    print("\n----------------------------------------------")
    print("[RETRIEVAL INSPECTION: ADAPTIVE HYBRID SEARCH]:")
    print("----------------------------------------------")
    if not reranked_nodes:
        print("No valid contexts retrieved (Guardrail active).")
    else:
        for i, node in enumerate(reranked_nodes):
            score = node.score if hasattr(node, 'score') and node.score is not None else 0.0
            print(f"\n[Reranked Chunk {i+1} | Score: {score:.4f}]")
            #print(node.node.get_content()[:400] + "...")
            print(node.node.get_content() + "...")

    # print("\n----------------------------------------------")
    print("[Final Generated LLM Response (Adaptive RAG)]:")
    print("----------------------------------------------")
    print(response)

if __name__ == "__main__":
    main()