import os
import math
import time
from src.pipeline.retriever.bm25 import BM25Retriever
from src.pipeline.retriever.dense import DenseRetriever

class HybridRetriever:
    def __init__(self, 
                 corpus_path: str = "data/retrieval/retrieval_corpus.jsonl",
                 bm25_cache_path: str = "data/retrieval/bm25_cache_segmented.pkl",
                 vector_db_path: str = "data/vector_store",
                 collection_name: str = "legal_articles",
                 embedding_provider: str = "local",
                 embedding_model_name: str = "BAAI/bge-m3",
                 embedding_api_url: str = "http://localhost:11434/api/embeddings",
                 embedding_api_key: str = ""):
        
        print("\n[Hybrid] Initializing Hybrid Retriever...", flush=True)
        
        # 1. Initialize BM25
        self.bm25_retriever = BM25Retriever(
            corpus_path=corpus_path,
            cache_path=bm25_cache_path,
            segmented=True,
            engine="pyvi",
            remove_stopwords=True
        )
        
        # 2. Initialize Dense (Cosine metric)
        self.dense_retriever = DenseRetriever(
            corpus_path=corpus_path,
            db_path=vector_db_path,
            collection_name=collection_name,
            provider=os.environ.get("DENSE_PROVIDER", embedding_provider),
            model_name=embedding_model_name,
            api_url=os.environ.get("DENSE_API_URL", embedding_api_url),
            api_key=os.environ.get("DENSE_API_KEY", embedding_api_key),
            auto_build=False
        )

        self.domain_doc_counts = {}
        for doc in self.bm25_retriever.corpus:
            chude = (doc.get("chude") or "").strip()
            if chude:
                self.domain_doc_counts[chude] = self.domain_doc_counts.get(chude, 0) + 1
        self.max_domain_count = max(self.domain_doc_counts.values(), default=1)
        
        print("[Hybrid] Hybrid Retriever initialized successfully.\n", flush=True)

    def _generic_title_penalty(self, doc: dict) -> float:
        title = (doc.get("law_title") or "").strip().lower()
        if not title:
            return 0.82
        generic_titles = {
            "nghị định",
            "thông tư",
            "luật",
            "quyết định",
            "nghị quyết",
            "nghị định quy định chi tiết một số điều",
            "nghị định quy định chi tiết thi hành một số điều",
            "thông tư hướng dẫn thi hành một số điều",
            "thông tư hướng dẫn thực hiện một số điều",
        }
        if title in generic_titles:
            return 0.72
        if len(title) < 24:
            return 0.90
        return 1.0

    def _generic_title_rescue(self, query: str, doc: dict) -> float:
        query_lower = query.lower()
        title = (doc.get("law_title") or "").strip().lower()
        demuc = (doc.get("demuc") or "").lower()
        chude = (doc.get("chude") or "").lower()
        citation = (doc.get("citation") or "").lower()

        generic_titles = {
            "nghị định",
            "thông tư",
            "luật",
            "quyết định",
            "nghị quyết",
            "nghị định quy định chi tiết một số điều",
            "nghị định quy định chi tiết thi hành một số điều",
            "thông tư hướng dẫn thi hành một số điều",
            "thông tư hướng dẫn thực hiện một số điều",
        }
        if title not in generic_titles:
            return 1.0

        rescue = 1.0
        if any(kw in query_lower for kw in ["thuế", "hóa đơn", "khấu trừ", "miễn thuế", "giảm thuế", "nợ thuế"]):
            if "quản lý thuế" in demuc or "thuế, phí, lệ phí" in chude:
                rescue = max(rescue, 1.20)
            if any(code in citation for code in ["126/2020/nđ-cp", "123/2020/nđ-cp", "80/2021/tt-btc", "78/2021/tt-btc"]):
                rescue = max(rescue, 1.30)

        if any(kw in query_lower for kw in ["doanh nghiệp nhỏ và vừa", "hỗ trợ tư vấn", "tư vấn viên", "ngân sách nhà nước"]):
            if "hỗ trợ doanh nghiệp nhỏ và vừa" in demuc:
                rescue = max(rescue, 1.25)
            if any(code in citation for code in ["80/2021/nđ-cp", "52/2023/tt-btc", "06/2022/tt-bkhđt"]):
                rescue = max(rescue, 1.35)

        if any(kw in query_lower for kw in ["lao động", "bằng cấp", "hợp đồng lao động", "kỷ luật"]):
            if "lao động" in chude:
                rescue = max(rescue, 1.18)
            if any(code in citation for code in ["12/2022/nđ-cp", "145/2020/nđ-cp", "45/2019/qh14"]):
                rescue = max(rescue, 1.28)

        return rescue

    def _domain_bias_penalty(self, doc: dict) -> float:
        chude = (doc.get("chude") or "").strip()
        if not chude:
            return 1.0
        count = self.domain_doc_counts.get(chude, 0)
        if count <= 0 or self.max_domain_count <= 1:
            return 1.0
        dominance = math.log1p(count) / math.log1p(self.max_domain_count)
        return max(0.84, 1.0 - 0.12 * dominance)

    def _apply_bias_penalty(self, query: str, documents: list[dict]) -> list[dict]:
        adjusted = []
        for doc in documents:
            item = doc.copy()
            bias_penalty = self._domain_bias_penalty(item)
            title_penalty = self._generic_title_penalty(item)
            title_rescue = self._generic_title_rescue(query, item)
            penalty = bias_penalty * title_penalty * title_rescue
            item["bias_penalty"] = penalty
            item["score"] = float(item.get("score", 0.0) * penalty)
            adjusted.append(item)
        adjusted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return adjusted

    def retrieve(self, query: str, top_k: int = 20, method: str = "rrf", rrf_k: int = 60, alpha: float = 0.5, domain_filter: list[str] = None) -> list[dict]:
        """
        Retrieves top_k documents using either:
        - 'rrf': Reciprocal Rank Fusion (with normalization to [0, 1] range)
        - 'linear': Linear Score Fusion (weighted sum of normalized scores)
        """
        start_time = time.time()
        print(f"=== Starting Hybrid Retrieval ({method.upper()}) for query: '{query}' ===", flush=True)
        
        # Retrieve twice the top_k to have a healthy candidate pool for fusion
        candidate_k = max(top_k * 2, 50)
        
        # BM25 Retrieval (Includes Probabilistic Gating inside BM25Retriever)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k, domain_filter=domain_filter)
        
        # NOTE: Apply Probabilistic Gating to Dense Results
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        if domain_filter:
            for doc in dense_results:
                doc_chude = doc.get("chude", "").lower()
                doc_title = doc.get("law_title", "").lower()
                if not any(d.lower() in doc_chude or d.lower() in doc_title for d in domain_filter):
                    doc["score"] *= 0.6  # Penalize out-of-domain by 40%
        dense_results = self._apply_bias_penalty(query, dense_results)
        bm25_results = self._apply_bias_penalty(query, bm25_results)
            
        if not bm25_results and not dense_results:
            print("[Hybrid] No documents retrieved by either BM25 or Dense search.", flush=True)
            return []

        # Print Top 3 BM25
        if bm25_results:
            print(f"[Hybrid] Top 3 BM25 Candidates:")
            for i, d in enumerate(bm25_results[:3], 1):
                title_str = f"{d.get('law_title', 'N/A')} | {d.get('article', 'N/A')}"
                print(f"  {i}. {title_str} (ID: {d['node_id']}) - Score: {d.get('score', 0):.4f}")

        # Print Top 3 Dense
        if dense_results:
            print(f"[Hybrid] Top 3 Dense Candidates:")
            for i, d in enumerate(dense_results[:3], 1):
                title_str = f"{d.get('law_title', 'N/A')} | {d.get('article', 'N/A')}"
                print(f"  {i}. {title_str} (ID: {d['node_id']}) - Score: {d.get('score', 0):.4f}")

        results = []
        if method.lower() == "rrf":
            # 1. Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            doc_details = {}
            
            # BM25 ranks
            for rank, doc in enumerate(bm25_results, 1):
                node_id = doc["node_id"]
                if node_id not in rrf_scores:
                    rrf_scores[node_id] = 0.0
                    doc_details[node_id] = doc
                rrf_scores[node_id] += 1.0 / (rrf_k + rank)
                doc_details[node_id]["bm25_rank"] = rank
                doc_details[node_id]["bm25_score"] = doc["score"]
                
            # Dense ranks
            for rank, doc in enumerate(dense_results, 1):
                node_id = doc["node_id"]
                if node_id not in rrf_scores:
                    rrf_scores[node_id] = 0.0
                    doc_details[node_id] = doc
                rrf_scores[node_id] += 1.0 / (rrf_k + rank)
                doc_details[node_id]["dense_rank"] = rank
                doc_details[node_id]["dense_score"] = doc["score"]
            
            # Normalize RRF scores so the maximum possible RRF score is 1.0.
            # Max possible score occurs when a document is rank 1 in both retrievers.
            # Max score = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 1)) = 2.0 / (rrf_k + 1)
            max_possible_rrf = 2.0 / (rrf_k + 1)
            
            # Sort by RRF score
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_docs[:top_k]
            
            for node_id, raw_rrf in top_candidates:
                doc = doc_details[node_id].copy()
                # Normalized score to range [0, 1]
                doc["score"] = float(raw_rrf / max_possible_rrf)
                doc["raw_rrf_score"] = raw_rrf
                results.append(doc)
                
            union_len = len(rrf_scores)
        else:
            # 2. Linear Score Fusion
            # Normalize BM25 scores to [0, 1] using Min-Max scaling
            bm25_min = min([d["score"] for d in bm25_results]) if bm25_results else 0.0
            bm25_max = max([d["score"] for d in bm25_results]) if bm25_results else 1.0
            bm25_denom = (bm25_max - bm25_min) if (bm25_max - bm25_min) > 0 else 1.0
            
            # Normalize Dense scores to [0, 1]
            dense_min = min([d["score"] for d in dense_results]) if dense_results else 0.0
            dense_max = max([d["score"] for d in dense_results]) if dense_results else 1.0
            dense_denom = (dense_max - dense_min) if (dense_max - dense_min) > 0 else 1.0
            
            combined_scores = {}
            doc_details = {}
            
            # Add BM25 normalized scores
            for rank, doc in enumerate(bm25_results, 1):
                node_id = doc["node_id"]
                norm_score = (doc["score"] - bm25_min) / bm25_denom
                combined_scores[node_id] = alpha * norm_score
                doc_details[node_id] = doc
                doc_details[node_id]["bm25_rank"] = rank
                doc_details[node_id]["bm25_score"] = doc["score"]
                doc_details[node_id]["bm25_norm_score"] = norm_score
                
            # Add Dense normalized scores
            for rank, doc in enumerate(dense_results, 1):
                node_id = doc["node_id"]
                norm_score = (doc["score"] - dense_min) / dense_denom
                
                if node_id not in combined_scores:
                    combined_scores[node_id] = 0.0
                    doc_details[node_id] = doc
                combined_scores[node_id] += (1.0 - alpha) * norm_score
                doc_details[node_id]["dense_rank"] = rank
                doc_details[node_id]["dense_score"] = doc["score"]
                doc_details[node_id]["dense_norm_score"] = norm_score

            # Sort by combined score
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_docs[:top_k]
            
            for node_id, score in top_candidates:
                doc = doc_details[node_id].copy()
                doc["score"] = float(score)
                results.append(doc)
                
            union_len = len(combined_scores)
            
        bm25_ids = set([d["node_id"] for d in bm25_results])
        dense_ids = set([d["node_id"] for d in dense_results])
        intersection_len = len(bm25_ids & dense_ids)
        
        print(f"[Hybrid] BM25 candidates: {len(bm25_results)}, Dense candidates: {len(dense_results)}", flush=True)
        print(f"[Hybrid] Intersection: {intersection_len}, Union: {union_len}", flush=True)
        
        # Print Top 3 Hybrid
        if results:
            print(f"[Hybrid] Top 3 Fusion ({method.upper()}) Results:")
            for i, d in enumerate(results[:3], 1):
                title_str = f"{d.get('law_title', 'N/A')} | {d.get('article', 'N/A')}"
                print(f"  {i}. {title_str} (ID: {d['node_id']}) - Score: {d.get('score', 0):.4f}")
        
        print(f"[Hybrid] Fusion completed in {time.time() - start_time:.4f}s. Returning top {len(results)} results.", flush=True)
        return results

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Testing Hybrid Retriever ===")
    
    hybrid_retriever = HybridRetriever(
        embedding_provider="local",
        embedding_model_name="BAAI/bge-m3"
    )
    
    test_query = "Nếu công ty cung cấp dịch vụ trung gian để khách hàng đăng bán sản phẩm sử dụng nhãn hiệu trùng với nhãn hiệu được bảo hộ thì công ty sẽ bị xác định hành vi xâm phạm và chịu trách nhiệm pháp lý như thế nào?"
    
    # 1. Test RRF Normalized
    print("\n--- Running RRF Fusion ---")
    results_rrf = hybrid_retriever.retrieve(test_query, top_k=5, method="rrf")
    print("\n=== Top 5 Hybrid RRF Results (Normalized) ===")
    for rank, doc in enumerate(results_rrf, 1):
        bm25_rank = doc.get("bm25_rank", "N/A")
        dense_rank = doc.get("dense_rank", "N/A")
        print(f"Rank {rank} | Normalized RRF Score: {doc['score']:.4f} | Node ID: {doc['node_id']}")
        print(f"BM25 Rank: {bm25_rank} | Dense Rank: {dense_rank}")
        print(f"Title: {doc['law_title']}")
        print(f"Article: {doc['article']}")
        print(f"Content snippet: {doc['content'][:150]}...")
        print("-" * 50)
        
    # 2. Test Linear Fusion
    print("\n--- Running Linear Score Fusion ---")
    results_linear = hybrid_retriever.retrieve(test_query, top_k=5, method="linear", alpha=0.5)
    print("\n=== Top 5 Hybrid Linear Fusion Results ===")
    for rank, doc in enumerate(results_linear, 1):
        bm25_rank = doc.get("bm25_rank", "N/A")
        dense_rank = doc.get("dense_rank", "N/A")
        print(f"Rank {rank} | Linear Score: {doc['score']:.4f} | Node ID: {doc['node_id']}")
        print(f"BM25 Rank: {bm25_rank} | Dense Rank: {dense_rank}")
        print(f"Title: {doc['law_title']}")
        print(f"Article: {doc['article']}")
        print(f"Content snippet: {doc['content'][:150]}...")
        print("-" * 50)
