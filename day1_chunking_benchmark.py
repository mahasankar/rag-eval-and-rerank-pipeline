"""
Day 1: Advanced Chunking Strategy Benchmarking (Final Clean Version)
- PyMuPDF (fitz) for pristine PDF extraction (No garbage font streams)
- Batched Vector Insertion (Prevents 8GB RAM OOM crashes)
- Persistent Chroma Storage (Instant subsequent loads)
"""

# C:\Users\mahad\venv\Scripts\activate.bat

import os
import time
import pymupdf  # PyMuPDF for clean text extraction
import chromadb

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.groq import Groq

# 1. Configure lightweight local embedding model (HuggingFace BGE)
print("Initializing local embedding model (HuggingFace BGE)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Configure local open-source LLM (Ollama running Llama 3.2 1B with safe context limit)
#print("Connecting to local Ollama LLM engine (Llama 3.2 1B)...")
#Settings.llm = Ollama(
    #model="llama3.2:1b", 
    #request_timeout=120.0,
    #additional_kwargs={"num_ctx": 2048}
#)

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
Settings.chunk_size = 512

# 3. Load PDF Document cleanly using PyMuPDF (Bypasses all font-encoding noise)
pdf_filename = "cracking_coding_interview.pdf" 

if not os.path.exists(pdf_filename):
    print(f"\n[!] Error: Could not find '{pdf_filename}' in the current directory.")
    exit(1)

print(f"Extracting clean text from {pdf_filename} using PyMuPDF...")
start_time = time.time()

doc = pymupdf.open(pdf_filename)
llama_documents = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    if text.strip():  # Only keep pages with actual text
        llama_documents.append(Document(text=text, metadata={"page_number": page_num + 1}))

print(f"Successfully extracted {len(llama_documents)} clean text pages in {time.time() - start_time:.2f} seconds.")

def batched_vector_insert(index, nodes, batch_size=200):
    """Inserts nodes into Chroma in safe batches to protect 8GB RAM."""
    total_nodes = len(nodes)
    for i in range(0, total_nodes, batch_size):
        batch = nodes[i : i + batch_size]
        index.insert_nodes(batch)
        print(f"Progress: Embedded & inserted {min(i + batch_size, total_nodes)}/{total_nodes} nodes...", end="\r")
    print()

def benchmark_sentence_window():
    print("\n==============================================")
    print("--- Strategy 1: Sentence-Window Chunking ---")
    print("==============================================")
    
    start_time = time.time()
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(llama_documents)
    print(f"[Timing] Generated {len(nodes)} sentence nodes in {time.time() - start_time:.2f} seconds.")

    print("[Timing] Connecting to Persistent Chroma Vector Store...")
    db = chromadb.PersistentClient(path="./chroma_db_sentence_window")
    chroma_collection = db.get_or_create_collection("sentence_window_full")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    embed_start = time.time()
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    if chroma_collection.count() == 0:
        print("[Timing] First run: Embedding sentence nodes in safe batches...")
        batched_vector_insert(index, nodes, batch_size=200)
    else:
        print(f"[Timing] Loaded existing vector store from disk instantly! ({chroma_collection.count()} items)")
    print(f"[Timing] Index ready in {time.time() - embed_start:.2f} seconds.")
    
    query_engine = index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
    )
    
    query = "What are the key tips for tackling technical interview questions?"
    print(f"\nQuery: '{query}'")
    
    query_start = time.time()
    response = query_engine.query(query)
    print(f"[Timing] Query executed in {time.time() - query_start:.2f} seconds.")
    
    print("\n----------------------------------------------")
    print("[RETRIEVAL INSPECTION] Clean Chunks Found in PDF:")
    print("----------------------------------------------")
    for i, source_node in enumerate(response.source_nodes):
        score_str = f"Score: {source_node.score:.4f}" if source_node.score else "Score: N/A"
        print(f"\n[Retrieved Chunk {i+1} | {score_str}]")
        print(source_node.node.get_content() + "...")
    print("----------------------------------------------")

    print(f"\n[Final Generated LLM Response]:\n{response}\n")

def benchmark_hierarchical_parent_child():
    print("\n==============================================")
    print("--- Strategy 2: Hierarchical Parent-Child Chunking ---")
    print("==============================================")
    
    start_time = time.time()
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512]
    )
    nodes = node_parser.get_nodes_from_documents(llama_documents)
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"[Timing] Generated hierarchical tree: Total nodes = {len(nodes)}, Leaf nodes = {len(leaf_nodes)} in {time.time() - start_time:.2f} seconds.")

    print("[Timing] Connecting to Persistent Chroma Vector Store...")
    db = chromadb.PersistentClient(path="./chroma_db_hierarchical")
    chroma_collection = db.get_or_create_collection("hierarchical_full")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    embed_start = time.time()
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    if chroma_collection.count() == 0:
        print("[Timing] First run: Embedding leaf nodes in safe batches...")
        batched_vector_insert(index, leaf_nodes, batch_size=200)
    else:
        print(f"[Timing] Loaded existing hierarchical vector store from disk instantly! ({chroma_collection.count()} items)")
    print(f"[Timing] Index ready in {time.time() - embed_start:.2f} seconds.")
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    query = "What are the key tips for tackling technical interview questions?"
    print(f"\nQuery: '{query}'")
    
    query_start = time.time()
    response = query_engine.query(query)
    print(f"[Timing] Query executed in {time.time() - query_start:.2f} seconds.")
    
    print("\n----------------------------------------------")
    print("[RETRIEVAL INSPECTION] Clean Chunks Found in PDF:")
    print("----------------------------------------------")
    for i, source_node in enumerate(response.source_nodes):
        score_str = f"Score: {source_node.score:.4f}" if source_node.score else "Score: N/A"
        print(f"\n[Retrieved Chunk {i+1} | {score_str}]")
        print(source_node.node.get_content() + "...")
    print("----------------------------------------------")

    print(f"\n[Final Generated LLM Response]:\n{response}\n")

if __name__ == "__main__":
    total_start = time.time()
    benchmark_sentence_window()
    benchmark_hierarchical_parent_child()
    print(f"\n==============================================")
    print(f"Benchmark run completed in {time.time() - total_start:.2f} seconds total!")
    print(f"==============================================")