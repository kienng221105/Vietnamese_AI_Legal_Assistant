import time
import os
import requests
import threading
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", api_url: str = None):
        self.api_url = api_url or os.environ.get("RERANKER_API_URL")
        self._lock = threading.Lock()
        
        if self.api_url:
            print(f"\n[Reranker] Initializing Remote Reranker using API: {self.api_url}", flush=True)
            self.model = "remote"
            return
            
        print(f"\n[Reranker] Initializing Cross-Encoder Reranker using '{model_name}' on '{device}'...", flush=True)
        start_time = time.time()
        try:
            # Load local_files_only first, fallback to download
            try:
                self.model = CrossEncoder(model_name, device=device, max_length=512, local_files_only=True)
            except Exception:
                self.model = CrossEncoder(model_name, device=device, max_length=512)
            print(f"[Reranker] Model loaded successfully in {time.time() - start_time:.2f}s.", flush=True)
        except Exception as e:
            print(f"[Reranker] Failed to load model '{model_name}': {e}", flush=True)
            print("[Reranker] Falling back to dummy reranker (returns input scores unchanged).", flush=True)
            self.model = None

    @staticmethod
    def _fallback_documents(documents: list[dict]) -> list[dict]:
        fallback_docs = []
        for doc in documents:
            item = doc.copy()
            original_score = float(item.get("score", 0.0))
            if "hybrid_score" not in item:
                item["hybrid_score"] = original_score
            item["rerank_score"] = original_score
            fallback_docs.append(item)
        return sorted(fallback_docs, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        """
        Reranks a list of candidate documents against the query.
        Each document dict must contain 'content' and preferably 'node_id'.
        """
        if not documents:
            return []
            
        if not self.model:
            # Fallback: Sort by existing score or return as-is
            print("[Reranker] Model not initialized. Returning documents sorted by original score.", flush=True)
            return self._fallback_documents(documents)[:top_k]

        start_time = time.time()
        print(f"[Reranker] Reranking {len(documents)} documents...", flush=True)
        
        if self.model == "remote":
            # Remote API Inference
            try:
                payload = {
                    "query": query,
                    "documents": [{"id": doc.get("node_id", str(i)), "text": doc["content"]} for i, doc in enumerate(documents)],
                    "top_k": top_k
                }
                headers = {"ngrok-skip-browser-warning": "1"}
                with self._lock:
                    response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                
                api_results = response.json()
                
                # Map scores back to original documents
                score_map = {item["id"]: item["score"] for item in api_results}
                
                # Filter and update only documents returned by the API
                reranked_docs = []
                for i, doc in enumerate(documents):
                    doc_id = doc.get("node_id", str(i))
                    if doc_id in score_map:
                        score = float(score_map[doc_id])
                        doc["rerank_score"] = score
                        if "hybrid_score" not in doc:
                            doc["hybrid_score"] = doc.get("score", 0.0)
                        doc["score"] = score
                        reranked_docs.append(doc)
                        
                reranked_docs = sorted(reranked_docs, key=lambda x: x["rerank_score"], reverse=True)
                print(f"[Reranker] Remote API Rerank completed in {time.time() - start_time:.4f}s.", flush=True)
                return reranked_docs
                
            except Exception as e:
                print(f"[Reranker] Remote API Error: {e}. Falling back to original scores.", flush=True)
                return self._fallback_documents(documents)[:top_k]
                
        # Local Inference
        # Prepare inputs: list of [query, text] pairs
        pairs = [[query, doc["content"]] for doc in documents]
        
        try:
            with self._lock:
                scores = self.model.predict(pairs)
            
            # Update scores in documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                if "hybrid_score" not in doc:
                    doc["hybrid_score"] = doc.get("score", 0.0)
                doc["score"] = float(score)
                
            # Sort by rerank score descending
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            print(f"[Reranker] Rerank completed in {time.time() - start_time:.4f}s. Returning top {top_k}.", flush=True)
            return reranked_docs[:top_k]
        except Exception as e:
            print(f"[Reranker] Error during reranking: {e}", flush=True)
            return self._fallback_documents(documents)[:top_k]
