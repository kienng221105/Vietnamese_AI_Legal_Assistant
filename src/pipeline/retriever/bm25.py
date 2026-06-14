import os
import json
import pickle
import time
from rank_bm25 import BM25Okapi
from src.utils.text_cleaner import TextCleaner

class BM25Retriever:
    def __init__(self, 
                 corpus_path: str = "data/retrieval/retrieval_corpus.jsonl", 
                 cache_path: str = "data/retrieval/bm25_cache_segmented.pkl", 
                 segmented: bool = True,
                 engine: str = "pyvi",
                 remove_stopwords: bool = True):
        self.corpus_path = corpus_path
        self.cache_path = cache_path
        self.segmented = segmented
        self.engine = engine
        self.remove_stopwords = remove_stopwords
        self.corpus = []
        self.bm25 = None
        
        print(f"\n[BM25] Initializing BM25 Retriever...")
        print(f"[BM25] Corpus path: {os.path.abspath(self.corpus_path)}")
        print(f"[BM25] Cache path: {os.path.abspath(self.cache_path)}")
        print(f"[BM25] Word segmentation (engine: {self.engine}) enabled: {self.segmented}")
        print(f"[BM25] Stopwords filtering enabled: {self.remove_stopwords}")
        
        self.load_index()

    def load_index(self):
        # 1. Try loading from pickle cache first
        if os.path.exists(self.cache_path):
            start_time = time.time()
            print(f"[BM25] Found cached index at {self.cache_path}. Loading cache...", flush=True)
            try:
                with open(self.cache_path, "rb") as f:
                    self.corpus, self.bm25 = pickle.load(f)
                print(f"[BM25] Cache loaded successfully in {time.time() - start_time:.2f}s. Total documents: {len(self.corpus)}", flush=True)
                return
            except Exception as e:
                print(f"[BM25] Error loading cache: {e}. Rebuilding index...", flush=True)

        # 2. If no cache or load failed, build it from corpus file
        start_time = time.time()
        print(f"[BM25] Loading and segmenting corpus from {self.corpus_path}...", flush=True)
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        documents_tokens = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                self.corpus.append(doc)
                
                # Use retrieval_text as the search field, but BM25F style boost for law_title
                law_title = doc.get("law_title", "")
                base_text = doc.get("retrieval_text", "")
                text = (law_title + " ") * 3 + base_text
                
                tokens = TextCleaner.tokenize(
                    text, 
                    lowercase=True, 
                    segmented=self.segmented, 
                    engine=self.engine, 
                    remove_stopwords=self.remove_stopwords
                )
                documents_tokens.append(tokens)
                
                if (idx + 1) % 15000 == 0:
                    print(f"[BM25] Loaded and tokenized {idx + 1} documents...", flush=True)

        print(f"[BM25] Finished loading & segmenting {len(self.corpus)} documents in {time.time() - start_time:.2f}s.", flush=True)
        
        print(f"[BM25] Building BM25 index...", flush=True)
        bm25_start = time.time()
        self.bm25 = BM25Okapi(documents_tokens)
        print(f"[BM25] BM25 index built in {time.time() - bm25_start:.2f}s.", flush=True)
        
        # Save to cache
        print(f"[BM25] Saving index and corpus to cache at {self.cache_path}...", flush=True)
        cache_start = time.time()
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump((self.corpus, self.bm25), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[BM25] Cache saved successfully in {time.time() - cache_start:.2f}s.", flush=True)
        except Exception as e:
            print(f"[BM25] Failed to save cache: {e}", flush=True)

    def retrieve(self, query: str, top_k: int = 20, domain_filter: list[str] = None) -> list[dict]:
        """
        Retrieves the top K documents for the given query.
        Returns a list of dicts with doc details and BM25 score.
        If domain_filter is provided, only documents matching one of the domains in 'chude' or 'law_title' are returned.
        """
        start_time = time.time()
        print(f"\n[BM25-Search] Query: '{query}'", flush=True)
        if domain_filter:
            print(f"[BM25-Search] Applying domain pre-filter: {domain_filter}", flush=True)
        
        # Tokenize query using the same settings
        tokenized_query = TextCleaner.tokenize(
            query, 
            lowercase=True, 
            segmented=self.segmented, 
            engine=self.engine, 
            remove_stopwords=self.remove_stopwords
        )
        print(f"[BM25-Search] Tokenized query: {tokenized_query}", flush=True)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and get top K indices
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                final_score = score
                if domain_filter:
                    doc_chude = self.corpus[idx].get("chude", "").lower()
                    doc_title = self.corpus[idx].get("law_title", "").lower()
                    
                    match = False
                    for d in domain_filter:
                        d_lower = d.lower()
                        if d_lower in doc_chude or d_lower in doc_title:
                            match = True
                            break
                    if not match:
                        # PROBABILISTIC GATING: Penalize by 50% instead of hard cut
                        final_score = score * 0.5
                        
                results.append((final_score, idx))
        
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:top_k]
        
        retrieved_docs = []
        for score, idx in top_results:
            doc = self.corpus[idx].copy()
            doc["score"] = score
            retrieved_docs.append(doc)
            
        print(f"[BM25-Search] Found {len(results)} matches. Retained top {len(retrieved_docs)} in {time.time() - start_time:.4f}s.", flush=True)
        return retrieved_docs

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Testing BM25 Retriever ===")
    retriever = BM25Retriever(segmented=True, engine="pyvi", remove_stopwords=True)
    
    # Run the user's trademark / liability query
    test_query = "Nếu công ty cung cấp dịch vụ trung gian để khách hàng đăng bán sản phẩm sử dụng nhãn hiệu trùng với nhãn hiệu được bảo hộ thì công ty sẽ bị xác định hành vi xâm phạm và chịu trách nhiệm pháp lý như thế nào?"
    results = retriever.retrieve(test_query, top_k=5)
    
    print("\n=== Top 5 BM25 Results ===")
    for rank, doc in enumerate(results, 1):
        print(f"Rank {rank} | Score: {doc['score']:.4f} | Node ID: {doc['node_id']}")
        print(f"Title: {doc['law_title']}")
        print(f"Article: {doc['article']}")
        print(f"Content snippet: {doc['content'][:150]}...")
        print("-" * 50)