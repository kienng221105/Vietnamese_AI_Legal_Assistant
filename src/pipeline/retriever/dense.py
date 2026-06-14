import os
import json
import time
import sys
import requests
import threading
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import sqlite3

# --- ChromaDB Compatibility Patch (Colab 0.5.x -> Local 0.6.3) ---
try:
    from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData
    orig_load = PersistentData.load_from_file

    def get_dimension_from_db(pickle_path):
        segment_dir = os.path.basename(os.path.dirname(pickle_path))
        db_path = os.path.join(os.path.dirname(os.path.dirname(pickle_path)), "chroma.sqlite3")
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.dimension 
                    FROM segments s 
                    JOIN collections c ON s.collection = c.id 
                    WHERE s.id = ?;
                """, (segment_dir,))
                row = cursor.fetchone()
                conn.close()
                if row and row[0] is not None:
                    return row[0]
            except Exception:
                pass
        return 1024

    def patched_load(filename):
        data = orig_load(filename)
        if isinstance(data, dict):
            dim = data.get("dimensionality")
            if dim is None:
                dim = get_dimension_from_db(filename)
            return PersistentData(
                dimensionality=dim,
                total_elements_added=data.get("total_elements_added", 0),
                id_to_label=data.get("id_to_label", {}),
                label_to_id=data.get("label_to_id", {}),
                id_to_seq_id=data.get("id_to_seq_id", {})
            )
        else:
            if data.dimensionality is None:
                data.dimensionality = get_dimension_from_db(filename)
        return data

    PersistentData.load_from_file = patched_load
except Exception as e:
    print(f"[Dense] Warning: Failed to apply ChromaDB compatibility patch: {e}", flush=True)


class APIEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that calls an external open-source API
    (Ollama, vLLM, or Hugging Face Inference API) to generate embeddings.
    """
    def __init__(self, api_url: str, model_name: str, api_key: str = None, provider: str = "ollama"):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.provider = provider.lower()
        self.is_e5 = "e5" in model_name.lower()
        self.timeout = int(os.environ.get("DENSE_API_TIMEOUT", "45"))
        self.expected_dim = self._infer_dimension(model_name)
        # Mode flag: "passage" for indexing, "query" for retrieval
        self._mode = "passage"
        print(f"[Embedding-API] Initialized API Embedding Client. URL: {self.api_url}, Provider: {self.provider}, E5: {self.is_e5}", flush=True)

    @staticmethod
    def _infer_dimension(model_name: str) -> int:
        name = (model_name or "").lower()
        if "bge-m3" in name:
            return 1024
        if "bge-large" in name:
            return 1024
        if "bge-base" in name:
            return 768
        if "bge-small" in name:
            return 384
        return int(os.environ.get("DENSE_EMBEDDING_DIM", "1024"))

    def set_mode(self, mode: str):
        """Set embedding mode: 'passage' for indexing, 'query' for retrieval."""
        self._mode = mode

    def __call__(self, input: Documents) -> Embeddings:
        processed_input = []
        for text in input:
            if self.is_e5:
                processed_input.append(f"{self._mode}: {text}")
            else:
                processed_input.append(text)

        try:
            if self.provider == "ollama":
                embeddings = []
                for processed_text in processed_input:
                    response = requests.post(
                        self.api_url,
                        json={"model": self.model_name, "prompt": processed_text},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    embedding = response_json.get("embedding", response_json.get("data", [{}])[0].get("embedding", []))
                    embeddings.append(embedding)
                return embeddings

            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            if self.provider == "huggingface":
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": processed_input},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                if isinstance(response_json, list) and response_json and isinstance(response_json[0], list):
                    return response_json
                return [response_json]

            response = requests.post(
                self.api_url,
                headers=headers,
                json={"model": self.model_name, "input": processed_input},
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_json = response.json()
            data = response_json.get("data", [])
            if not data:
                raise ValueError("Embedding API returned empty data")
            return [item["embedding"] for item in data]
        except Exception as e:
            sample = processed_input[0][:50] if processed_input else ""
            print(f"[Embedding-API] Error calling embedding API for text: {sample}... Error: {e}", flush=True)
            return [[0.0] * self.expected_dim for _ in processed_input]

class LocalEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that loads a SentenceTransformer model locally.
    Enforces normalized embeddings for correct cosine similarity.
    Supports E5-style prefixing with mode switching (passage vs query).
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        import torch
        import gc
        print(f"[Embedding-Local] Loading local model '{model_name}' on CPU/GPU...", flush=True)
        
        # Force garbage collection before loading large model
        gc.collect()
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"[Embedding-Local] Model loaded in FP32 successfully.", flush=True)
        except RuntimeError as e:
            if "not enough memory" in str(e):
                print(f"[Embedding-Local] OOM in FP32 load. Trying minimal load...", flush=True)
                gc.collect()
                self.model = SentenceTransformer(model_name)
            else:
                raise
            
        self.is_e5 = "e5" in model_name.lower()
        # Mode flag: "passage" for indexing, "query" for retrieval
        self._mode = "passage"
        print(f"[Embedding-Local] Local model loaded successfully. E5 prefix: {self.is_e5}", flush=True)

    def set_mode(self, mode: str):
        """Set embedding mode: 'passage' for indexing, 'query' for retrieval."""
        self._mode = mode

    def __call__(self, input: Documents) -> Embeddings:
        processed_input = []
        for text in input:
            if self.is_e5:
                processed_input.append(f"{self._mode}: {text}")
            else:
                processed_input.append(text)
                
        embeddings = self.model.encode(
            processed_input, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).tolist()
        return embeddings

class DenseRetriever:
    def __init__(self, 
                 corpus_path: str = "data/retrieval/retrieval_corpus.jsonl", 
                 db_path: str = "data/vector_store",
                 collection_name: str = "legal_articles",
                 provider: str = "local",
                 model_name: str = "BAAI/bge-m3",
                 api_url: str = "http://localhost:11434/api/embeddings",
                 api_key: str = "",
                 auto_build: bool = False):
        
        self.corpus_path = corpus_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.provider = provider.lower()
        self.model_name = model_name
        self.corpus_lookup = self._load_corpus_lookup(corpus_path)
        
        print(f"\n[Dense] Initializing Dense Vector Retriever...", flush=True)
        print(f"[Dense] Provider: {self.provider}", flush=True)
        print(f"[Dense] Model name: {self.model_name}", flush=True)
        print(f"[Dense] DB Path: {os.path.abspath(self.db_path)}", flush=True)

        if self.provider == "api":
            self.embedding_fn = APIEmbeddingFunction(
                api_url=api_url, 
                model_name=model_name, 
                api_key=api_key,
                provider="ollama" if "11434" in api_url else "openai"
            )
        else:
            self.embedding_fn = LocalEmbeddingFunction(model_name=model_name)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Configure collection to use Cosine distance
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Thread lock to prevent ChromaDB (SQLite/hnswlib) Segfaults during concurrent queries
        self._query_lock = threading.Lock()
        
        doc_count = self.collection.count()
        print(f"[Dense] Current document count in collection '{self.collection_name}': {doc_count}", flush=True)
        
        if doc_count == 0 and auto_build:
            self.build_index()
        elif doc_count > 0:
            print(f"[Dense] Vector store is ready for querying.", flush=True)

    def _load_corpus_lookup(self, corpus_path: str) -> dict:
        lookup = {}
        if not os.path.exists(corpus_path):
            return lookup

        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    node_id = doc.get("node_id")
                    if node_id:
                        lookup[node_id] = doc
            print(f"[Dense] Loaded corpus metadata lookup with {len(lookup)} docs.", flush=True)
        except Exception as e:
            print(f"[Dense] Warning: failed to load corpus metadata lookup: {e}", flush=True)
        return lookup

    def build_index(self, max_docs: int = None):
        """
        Loads corpus and inserts embeddings into the Chroma vector database.
        Embedding function mode is set to 'passage' during indexing.
        Optionally limits indexing to `max_docs` for fast testing.
        """
        # Set mode to 'passage' for indexing
        self.embedding_fn.set_mode("passage")
        
        start_time = time.time()
        print(f"[Dense] Building dense index from {self.corpus_path}...", flush=True)
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        # Read corpus
        docs_to_index = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                docs_to_index.append(doc)
                if max_docs and len(docs_to_index) >= max_docs:
                    break

        print(f"[Dense] Loaded {len(docs_to_index)} documents from corpus file.", flush=True)
        
        # Batch insert to ChromaDB
        batch_size = 100
        total_docs = len(docs_to_index)
        
        print(f"[Dense] Starting vector indexing (mode=passage). Batch size: {batch_size}", flush=True)
        for i in range(0, total_docs, batch_size):
            batch = docs_to_index[i:i+batch_size]
            batch_start = time.time()
            
            ids = [item["node_id"] for item in batch]
            documents = [item.get("retrieval_text", "") for item in batch]
            
            # Store metadata
            metadatas = []
            for item in batch:
                metadatas.append({
                    "law_title": item.get("law_title", ""),
                    "article": item.get("article", ""),
                    "chapter": item.get("chapter", ""),
                    "demuc": item.get("demuc", ""),
                    "chude": item.get("chude", ""),
                    "citation": item.get("citation", ""),
                    "content": item.get("content", "")[:1000]
                })
            
            # Add to ChromaDB (calls embedding_fn with mode='passage')
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            elapsed = time.time() - batch_start
            print(f"[Dense] Indexed batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} (Docs {i} to {min(i + batch_size, total_docs)}) in {elapsed:.2f}s ({len(batch)/elapsed:.1f} docs/s)", flush=True)
            
        print(f"[Dense] Vector store construction completed in {time.time() - start_time:.2f}s. Total stored docs: {self.collection.count()}", flush=True)

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Queries the vector database for similar legal documents.
        Embedding function mode is set to 'query' during retrieval.
        """
        start_time = time.time()
        print(f"\n[Dense-Search] Query: '{query}'", flush=True)
        
        # Switch embedding function to query mode
        self.embedding_fn.set_mode("query")
        
        # ChromaDB will call embedding_fn on query text, which now prefixes "query: "
        with self._query_lock:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
        
        # Switch back to passage mode (for any subsequent indexing)
        self.embedding_fn.set_mode("passage")
        
        retrieved_docs = []
        if results and "ids" in results and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc_id, metadata, distance in zip(ids, metadatas, distances):
                # Cosine distance = 1 - cosine_similarity
                # Therefore: similarity score = 1.0 - distance
                score = 1.0 - distance
                full_doc = self.corpus_lookup.get(doc_id, {})
                
                retrieved_docs.append({
                    "node_id": doc_id,
                    "score": float(score),
                    "law_title": full_doc.get("law_title", metadata.get("law_title", "")),
                    "article": full_doc.get("article", metadata.get("article", "")),
                    "article_number": full_doc.get("article_number", metadata.get("article_number", "")),
                    "chapter": full_doc.get("chapter", metadata.get("chapter", "")),
                    "demuc": full_doc.get("demuc", metadata.get("demuc", "")),
                    "chude": full_doc.get("chude", metadata.get("chude", "")),
                    "citation": full_doc.get("citation", metadata.get("citation", "")),
                    "content": full_doc.get("content", metadata.get("content", "")),
                    "retrieval_text": full_doc.get("retrieval_text", "")
                })
                
        print(f"[Dense-Search] Retrieved {len(retrieved_docs)} vectors in {time.time() - start_time:.4f}s.", flush=True)
        return retrieved_docs

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Testing Dense Retriever ===")
    
    retriever = DenseRetriever(
        provider="local",
        model_name="BAAI/bge-m3",
        auto_build=False
    )
    
    # Index only the first 500 documents for fast testing
    if retriever.collection.count() == 0:
        retriever.build_index(max_docs=500)
        
    test_query = "Nếu công ty cung cấp dịch vụ trung gian để khách hàng đăng bán sản phẩm sử dụng nhãn hiệu trùng với nhãn hiệu được bảo hộ thì công ty sẽ bị xác định hành vi xâm phạm và chịu trách nhiệm pháp lý như thế nào?"
    results = retriever.retrieve(test_query, top_k=5)
    
    print("\n=== Top 5 Dense Results ===")
    for rank, doc in enumerate(results, 1):
        print(f"Rank {rank} | Cosine Similarity: {doc['score']:.4f} | Node ID: {doc['node_id']}")
        print(f"Title: {doc['law_title']}")
        print(f"Article: {doc['article']}")
        print(f"Content snippet: {doc['content'][:150]}...")
        print("-" * 50)
