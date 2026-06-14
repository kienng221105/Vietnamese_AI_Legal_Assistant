import os
import time
import re
import json
from src.pipeline.retriever.hybrid import HybridRetriever
from src.pipeline.retriever.query_expander import QueryExpander
from src.pipeline.graph.expander import GraphExpander
from src.pipeline.retriever.reranker import Reranker
from src.pipeline.retriever.taxonomy_router import TaxonomyRouter

class RAGPipeline:
    def __init__(self,
                 corpus_path: str = "data/retrieval/retrieval_corpus.jsonl",
                 bm25_cache_path: str = "data/retrieval/bm25_cache_segmented.pkl",
                 vector_db_path: str = "data/vector_store",
                 collection_name: str = "legal_articles",
                 embedding_provider: str = "local",
                 embedding_model_name: str = "BAAI/bge-m3",
                 reranker_model_name: str = "BAAI/bge-reranker-base",
                 reranker_api_url: str | None = None,
                 adjacency_path: str = None,
                 kg_corpus_path: str = None,
                 fast_submission: bool = False):
        
        print("\n==================================================", flush=True)
        print("    Initializing Master RAG Pipeline Orchestrator  ", flush=True)
        print("==================================================\n", flush=True)
        
        # 1. Initialize retrievers
        self.hybrid_retriever = HybridRetriever(
            corpus_path=corpus_path,
            bm25_cache_path=bm25_cache_path,
            vector_db_path=vector_db_path,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name
        )
        
        # 2. Initialize local graph expander (uses adjacency.pkl, no Neo4j needed)
        self.graph_expander = GraphExpander(
            adjacency_path=adjacency_path,
            corpus_path=kg_corpus_path or corpus_path
        )
        
        reranker_model = os.environ.get("RERANKER_MODEL_PATH") or reranker_model_name
        self.reranker = Reranker(
            model_name=reranker_model,
            api_url=reranker_api_url or os.environ.get("RERANKER_API_URL")
        )
        
        self.query_expander = QueryExpander()
        self.taxonomy_router = TaxonomyRouter()
        self.fast_submission = fast_submission
        
        print("\n[RAGPipeline] Orchestrator initialization complete.\n", flush=True)

    def classify_query_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["điều kiện", "tiêu chí", "yêu cầu", "được phép", "đáp ứng"]):
            return "ELIGIBILITY"
        elif any(k in q for k in ["trình tự", "thủ tục", "hồ sơ", "bước", "cách thức", "làm thế nào", "hướng dẫn"]):
            return "PROCEDURE"
        elif any(k in q for k in ["phạt", "vi phạm", "trách nhiệm", "tù", "chế tài"]):
            return "SANCTION"
        elif any(k in q for k in ["là gì", "thế nào là", "được hiểu là", "định nghĩa", "giải thích"]):
            return "DEFINITION"
        return "GENERAL"

    def get_article_types(self, text: str, title: str) -> set:
        types = set()
        content = (title + " " + text).lower()
        
        if any(k in content for k in ["được hiểu là", "là gì", "giải thích từ ngữ", "nghĩa là", "định nghĩa"]):
            types.add("DEFINITION")
            
        if any(k in content for k in ["đáp ứng điều kiện", "tiêu chí xác định", "điều kiện để", "tiêu chí"]):
            types.add("ELIGIBILITY")
            
        if any(k in content for k in ["trình tự", "thủ tục", "hồ sơ", "các bước", "quy trình"]):
            types.add("PROCEDURE")
            
        if any(k in content for k in ["xử phạt", "vi phạm", "chế tài", "hình sự", "trách nhiệm", "xử lý"]):
            types.add("SANCTION")
            
        # If it's heavily procedural, remove eligibility to prevent false positives
        if "PROCEDURE" in types and ("quy trình, thủ tục" in text[:100].lower() or "thủ tục" in title.lower()):
            if "ELIGIBILITY" in types:
                types.remove("ELIGIBILITY")
            
        if not types:
            types.add("GENERAL")
            
        return types

    def calculate_answerability_score(self, query: str, doc: dict) -> float:
        """
        Stage 3: Answer-Driven Scoring.
        Does this set of articles fully answer the question?
        Combines ontology-aware Role Prior with Intent-Type matching.
        """
        query_lower = query.lower()
        doc_text = doc.get("content", "").lower()
        law_title = doc.get("law_title", "").lower()
        
        score = 0.5  # Base score
        
        # --- 1. Role Logic (Intermediary vs Infringement) ---
        is_intermediary_query = any(k in query_lower for k in ["trung gian", "nền tảng", "sàn giao dịch", "thương mại điện tử", "mạng xã hội", "sàn tmđt"])
        is_infringement_query = any(k in query_lower for k in ["xâm phạm", "vi phạm", "xử phạt", "hình sự", "chế tài"])
        
        if is_intermediary_query:
            has_intermediary_role = any(k in doc_text or k in law_title for k in ["trung gian", "nền tảng", "sàn giao dịch", "thương mại điện tử", "cung cấp dịch vụ", "trách nhiệm"])
            score += 0.2 if has_intermediary_role else -0.2
                
        elif is_infringement_query:
            has_infringement_role = any(k in doc_text or k in law_title for k in ["xâm phạm", "vi phạm", "xử phạt", "hình sự", "chế tài"])
            if has_infringement_role: score += 0.2
            
        # --- 2. Intent-Type Matching (Answer Correctness Alignment) ---
        intent = self.classify_query_intent(query)
        doc_types = self.get_article_types(doc_text, law_title)
        
        if intent == "GENERAL":
            score += 0.1
        elif intent in doc_types:
            score += 0.3  # Exact answer match
        elif "DEFINITION" in doc_types and intent != "DEFINITION":
            score -= 0.3  # STRIKE: Heavily penalize definition if they ask for procedure/eligibility
            
        return max(0.0, min(1.0, score))

    def calculate_primary_law_boost(self, query: str, doc: dict) -> float:
        """
        Primary Law Boost (Grounding for policy-heavy queries).
        Boosts primary laws (Luật/Bộ luật) for general/policy questions.
        """
        query_lower = query.lower()
        law_title = doc.get("law_title", "").lower()
        
        score = 0.5 # Neutral base score
        
        is_policy_query = any(kw in query_lower for kw in ["khái niệm", "quy định chung", "chính sách", "là gì", "như thế nào", "quyền", "nghĩa vụ"])
        if is_policy_query:
            # Boost if it is a primary law (Luật/Bộ luật), not a secondary regulation (Nghị định/Thông tư)
            if "luật" in law_title and "nghị định" not in law_title and "thông tư" not in law_title:
                score += 0.2
                
        return max(0.0, min(1.0, score))

    @staticmethod
    def merge_candidate(base: dict, incoming: dict) -> dict:
        """Merge duplicate candidates without losing richer corpus metadata."""
        for key, value in incoming.items():
            if key == "score":
                base[key] = max(base.get(key, 0.0), value or 0.0)
            elif key == "source":
                if value and value not in base.get("source", ""):
                    base[key] = f"{base.get('source', '')}_and_{value}".strip("_")
            elif value and not base.get(key):
                base[key] = value
        return base

    @staticmethod
    def _candidate_domain_key(doc: dict) -> str:
        chude = (doc.get("chude") or "").strip()
        if chude:
            return chude
        demuc = (doc.get("demuc") or "").strip()
        if demuc:
            return demuc
        law_title = (doc.get("law_title") or "").strip()
        return law_title or "__unknown__"

    def _select_diverse_candidates(
        self,
        candidates: list[dict],
        limit: int,
        global_keep: int,
        per_domain_quota: int,
    ) -> list[dict]:
        ordered = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        if len(ordered) <= limit:
            return ordered

        selected = []
        seen = set()
        domain_counts: dict[str, int] = {}

        def add_candidate(doc: dict) -> bool:
            node_id = doc.get("node_id")
            if not node_id or node_id in seen or len(selected) >= limit:
                return False
            selected.append(doc)
            seen.add(node_id)
            domain_key = self._candidate_domain_key(doc)
            domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
            return True

        for doc in ordered[:global_keep]:
            add_candidate(doc)

        for doc in ordered:
            if len(selected) >= limit:
                break
            domain_key = self._candidate_domain_key(doc)
            if domain_counts.get(domain_key, 0) < per_domain_quota:
                add_candidate(doc)

        for doc in ordered:
            if len(selected) >= limit:
                break
            add_candidate(doc)

        return selected

    def _backfill_supporting_candidates(
        self,
        primary_results: list[dict],
        candidate_docs: list[dict],
        limit: int,
    ) -> list[dict]:
        if len(primary_results) >= limit:
            return primary_results[:limit]

        selected = []
        seen = set()
        selected_laws = set()

        for doc in primary_results:
            node_id = doc.get("node_id")
            if node_id and node_id not in seen:
                selected.append(doc)
                seen.add(node_id)
                law_key = (doc.get("law_title") or "", doc.get("citation") or "")
                selected_laws.add(law_key)

        candidates_sorted = sorted(candidate_docs, key=lambda x: x.get("score", 0.0), reverse=True)

        for doc in candidates_sorted:
            if len(selected) >= limit:
                break
            node_id = doc.get("node_id")
            if not node_id or node_id in seen:
                continue
            law_key = (doc.get("law_title") or "", doc.get("citation") or "")
            candidate_score = doc.get("score", 0.0)
            if law_key in selected_laws:
                if candidate_score >= 0.45:
                    selected.append(doc)
                    seen.add(node_id)
                continue
            if candidate_score >= 0.55:
                selected.append(doc)
                seen.add(node_id)
                selected_laws.add(law_key)

        return selected[:limit]

    @staticmethod
    def _law_key(doc: dict) -> tuple[str, str]:
        return (doc.get("citation", "") or "", doc.get("law_title", "") or "")

    def _select_results_with_law_coverage(self, scored_docs: list[dict], limit: int) -> list[dict]:
        if len(scored_docs) <= limit:
            return scored_docs

        ordered = sorted(
            scored_docs,
            key=lambda x: x.get("combined_score", x.get("score", 0.0)),
            reverse=True,
        )

        law_groups: dict[tuple[str, str], list[dict]] = {}
        law_scores: dict[tuple[str, str], float] = {}
        law_family_scores: dict[tuple[str, str], float] = {}
        for doc in ordered:
            law_key = self._law_key(doc)
            law_groups.setdefault(law_key, []).append(doc)
            law_scores[law_key] = max(
                law_scores.get(law_key, 0.0),
                doc.get("combined_score", doc.get("score", 0.0)),
            )

        for law_key, docs in law_groups.items():
            doc_scores = [doc.get("combined_score", doc.get("score", 0.0)) for doc in docs]
            top_score = doc_scores[0]
            top2_avg = sum(doc_scores[:2]) / min(2, len(doc_scores))
            top3_avg = sum(doc_scores[:3]) / min(3, len(doc_scores))
            law_family_scores[law_key] = 0.55 * top_score + 0.30 * top2_avg + 0.15 * top3_avg

        ranked_laws = sorted(
            law_groups.keys(),
            key=lambda lk: (law_family_scores[lk], law_scores[lk], len(law_groups[lk])),
            reverse=True,
        )

        selected = []
        seen = set()
        law_counts: dict[tuple[str, str], int] = {}

        def add_doc(doc: dict, max_per_law: int) -> bool:
            node_id = doc.get("node_id")
            if not node_id or node_id in seen:
                return False
            law_key = self._law_key(doc)
            if law_counts.get(law_key, 0) >= max_per_law:
                return False
            selected.append(doc)
            seen.add(node_id)
            law_counts[law_key] = law_counts.get(law_key, 0) + 1
            return True

        primary_family_score = law_family_scores[ranked_laws[0]]
        chosen_laws = []
        for law_key in ranked_laws:
            if law_family_scores[law_key] >= primary_family_score * 0.82 or not chosen_laws:
                chosen_laws.append(law_key)
            elif len(chosen_laws) < 2:
                chosen_laws.append(law_key)
            else:
                break

        # Pass 1: lock the strongest law families first.
        for idx, law_key in enumerate(chosen_laws):
            if len(selected) >= limit:
                break
            quota = 3 if idx == 0 else 2
            law_top = law_scores[law_key]
            threshold = 0.74 if idx == 0 else 0.82
            for doc in law_groups[law_key]:
                if len(selected) >= limit:
                    break
                score = doc.get("combined_score", doc.get("score", 0.0))
                if score >= law_top * threshold:
                    add_doc(doc, max_per_law=quota)

        # Pass 2: add the top article from nearby law families that remain close to the top family.
        for law_key in ranked_laws:
            if len(selected) >= limit:
                break
            if law_key in chosen_laws:
                continue
            if law_family_scores[law_key] >= primary_family_score * 0.72:
                add_doc(law_groups[law_key][0], max_per_law=1)

        # Pass 3: fill the remaining slots by global score, but keep a soft cap per law.
        for doc in ordered:
            if len(selected) >= limit:
                break
            add_doc(doc, max_per_law=3)

        for law_key, docs in law_groups.items():
            for doc in docs:
                doc["law_family_score"] = law_family_scores[law_key]

        return selected[:limit]

    def prepare_candidates(
        self,
        query: str,
        hybrid_k: int = 150,
        max_hops: int = 1,
        fusion_method: str = "rrf",
    ) -> dict:
        if self.fast_submission:
            queries = [query]
        else:
            queries = self.query_expander.get_query_variations(query)
        domain_filter = self.taxonomy_router.route(query)
        if domain_filter:
            print(f"[Taxonomy Router] Routing query to specific domains: {domain_filter}", flush=True)

        seeds_map = {}
        for idx, q in enumerate(queries):
            print(f"[RAGPipeline] Running retrieval for query variation {idx}: '{q}'", flush=True)
            variation_seeds = self.hybrid_retriever.retrieve(
                query=q,
                top_k=hybrid_k,
                method=fusion_method,
                domain_filter=domain_filter,
            )
            weight = 1.0 if idx == 0 else 0.9

            for s in variation_seeds:
                nid = s["node_id"]
                weighted_score = s["score"] * weight
                if nid not in seeds_map:
                    s["source"] = "hybrid_seed"
                    s["score"] = weighted_score
                    seeds_map[nid] = s
                else:
                    s["score"] = weighted_score
                    seeds_map[nid] = self.merge_candidate(seeds_map[nid], s)

        seeds = list(seeds_map.values())
        print(f"[RAGPipeline] Total unique seed candidates retrieved: {len(seeds)}", flush=True)

        if not seeds:
            return {
                "queries": queries,
                "domain_filter": domain_filter,
                "seeds": [],
                "top_seeds": [],
                "neighbors": [],
                "candidate_docs": [],
                "candidates_to_rerank": [],
            }

        seeds.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_seed_limit = 10 if self.fast_submission else 20
        top_seeds = seeds[:top_seed_limit]

        seed_ids = [s["node_id"] for s in top_seeds]
        raw_neighbors = self.graph_expander.expand_seeds(seed_ids=seed_ids, max_hops=max_hops)

        neighbors = []
        if domain_filter and raw_neighbors:
            for n in raw_neighbors:
                n_chude = n.get("chude", "").lower()
                n_title = n.get("law_title", "").lower()
                if not any(d.lower() in n_chude or d.lower() in n_title for d in domain_filter):
                    n["score"] = n.get("score", 0.5) * 0.7
                neighbors.append(n)
        else:
            neighbors = raw_neighbors

        candidates_map = {}
        for doc in seeds:
            candidates_map[doc["node_id"]] = doc

        for doc in neighbors:
            nid = doc["node_id"]
            if nid not in candidates_map:
                candidates_map[nid] = doc
            else:
                candidates_map[nid] = self.merge_candidate(candidates_map[nid], doc)

        candidate_docs = list(candidates_map.values())
        candidate_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        print(
            f"[RAGPipeline] Stage 3: Merged candidates. Seeds: {len(seeds)}, Neighbors: {len(neighbors)}, Unique Pool: {len(candidate_docs)}",
            flush=True,
        )

        candidates_to_rerank = list(candidate_docs)
        candidate_limit = 24 if self.fast_submission else 54
        if len(candidates_to_rerank) > candidate_limit:
            global_keep = 12 if self.fast_submission else 28
            per_domain_quota = 1 if self.fast_submission else 3
            candidates_to_rerank = self._select_diverse_candidates(
                candidates=candidates_to_rerank,
                limit=candidate_limit,
                global_keep=global_keep,
                per_domain_quota=per_domain_quota,
            )
            print(
                f"[RAGPipeline] Capped candidate pool to {len(candidates_to_rerank)} with per-domain quota={per_domain_quota}.",
                flush=True,
            )

        candidates_to_rerank = sorted(candidates_to_rerank, key=lambda x: x.get("score", 0.0), reverse=True)
        return {
            "queries": queries,
            "domain_filter": domain_filter,
            "seeds": seeds,
            "top_seeds": top_seeds,
            "neighbors": neighbors,
            "candidate_docs": candidate_docs,
            "candidates_to_rerank": candidates_to_rerank,
        }


    def retrieve_context(self, 
                         query: str, 
                         hybrid_k: int = 150, 
                         max_hops: int = 1, 
                         final_k: int = 10,
                         fusion_method: str = "rrf",
                         skip_rerank: bool = False) -> list[dict]:
        """
        Runs the full multi-stage RAG retrieval:
        1. Query Expansion (Synonym or LLM-based)
        2. Multi-query Hybrid Search (BM25 + Dense) -> Combined Seeds
        3. Knowledge Graph expansion -> Neighbors
        4. Merge & Deduplicate candidates
        5. Cross-Encoder rerank -> Top final_k
        """
        overall_start = time.time()
        print(f"\n[RAGPipeline] Processing query: '{query}'", flush=True)
        
        candidate_state = self.prepare_candidates(
            query=query,
            hybrid_k=hybrid_k,
            max_hops=max_hops,
            fusion_method=fusion_method,
        )
        seeds = candidate_state["seeds"]
        candidates_to_rerank = candidate_state["candidates_to_rerank"]

        if not seeds:
            print("[RAGPipeline] Stage 1 returned 0 seed candidates. Aborting.", flush=True)
            return []

        if skip_rerank:
            print("[RAGPipeline] Skipping Cross-Encoder Rerank as requested. Returning Hybrid + Graph candidates directly.", flush=True)
            return candidates_to_rerank[:final_k]
            
        # --- Stage 4: Cross-Encoder Rerank with Hybrid Score-Aware Fusion ---
        rerank_top_k = len(candidates_to_rerank) if not self.fast_submission else min(len(candidates_to_rerank), 16)
        scored_docs = self.reranker.rerank(
            query=query,
            documents=candidates_to_rerank,
            top_k=rerank_top_k
        )
        
        # Normalize reranker scores and fuse with hybrid scores
        if scored_docs:
            rerank_scores = [doc.get("rerank_score", 0.0) for doc in scored_docs]
            min_rerank = min(rerank_scores)
            max_rerank = max(rerank_scores)
            rerank_denom = (max_rerank - min_rerank) if (max_rerank - min_rerank) > 0.0 else 1.0
            
            for doc in scored_docs:
                # 1. Normalize rerank score to [0, 1]
                r_score = doc.get("rerank_score", 0.0)
                r_norm = (r_score - min_rerank) / rerank_denom

                # 2. Hybrid score (already in range [0, 1])
                h_norm = doc.get("hybrid_score", 0.0)
                answerability_score = self.calculate_answerability_score(query, doc)
                primary_law_score = self.calculate_primary_law_boost(query, doc)
                bias_penalty = doc.get("bias_penalty", 1.0)

                # 3. Stage-Aware Prior
                source = doc.get("source", "")
                if "hybrid_seed" in source and "graph_neighbor" in source:
                    stage_prior = 0.08
                elif "hybrid_seed" in source:
                    stage_prior = 0.05
                else:
                    stage_prior = 0.0

                combined_score = (
                    0.60 * r_norm
                    + 0.18 * h_norm
                    + 0.14 * answerability_score
                    + 0.08 * primary_law_score
                    + stage_prior
                ) * bias_penalty

                doc["score"] = combined_score
                doc["combined_score"] = combined_score
                doc["rerank_score_norm"] = r_norm
                doc["answerability_score"] = answerability_score
                doc["primary_law_score"] = primary_law_score
                doc["bias_penalty"] = bias_penalty
                
            final_results = self._select_results_with_law_coverage(scored_docs, final_k)
        else:
            final_results = []
            
        print(f"[RAGPipeline] Pipeline query finished in {time.time() - overall_start:.4f}s. Retrieved {len(final_results)} results.\n", flush=True)
        return final_results

    def close(self):
        pass  # No external connections to close

    @staticmethod
    def extract_law_id(citation: str, law_title: str = "") -> str:
        # Extracts patterns like "04/2017/QH14" or "80/2021/NĐ-CP" from citation
        if citation:
            match = re.search(r'(?:số\s+)?([0-9a-zA-Z]+/[0-9]{4}/[0-9a-zA-Z\-Đđ]+)', citation, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
        # Fallback Hardcode Mapping for messy corpus without IDs in citation
        law_title_lower = law_title.lower()
        HARDCODE_MAPPING = {
            "bộ luật lao động": "45/2019/QH14",
            "luật doanh nghiệp": "59/2020/QH14",
            "luật quản lý thuế": "38/2019/QH14",
            "luật hình sự": "100/2015/QH13",
            "an toàn, vệ sinh lao động": "84/2015/QH13",
            "quyền lợi người tiêu dùng": "59/2010/QH12",
            "luật thương mại": "36/2005/QH11",
            "luật dân sự": "91/2015/QH13",
            "luật hợp tác xã": "54/2012/QH13",
            "hỗ trợ doanh nghiệp nhỏ và vừa": "04/2017/QH14",
            "chuyển giao công nghệ": "08/2017/QH14",
            "trách nhiệm bồi thường": "10/2017/QH14",
            "hải quan": "54/2014/QH13",
            "chứng khoán": "54/2019/QH14",
            "hôn nhân và gia đình": "52/2014/QH13",
            "luật quảng cáo": "16/2012/QH13",
            "kinh doanh bảo hiểm": "08/2022/QH15",
            "đăng ký doanh nghiệp": "01/2021/NĐ-CP",
            "xử phạt vi phạm hành chính về thuế, hóa đơn": "125/2020/NĐ-CP",
        }
        
        for key, value in HARDCODE_MAPPING.items():
            if key in law_title_lower:
                return value
                
        return ""

    @staticmethod
    def format_submission(question: str, results: list[dict], top_k: int = 5) -> dict:
        """
        Formats the retrieval results into the Zalo AI Challenge submission format.
        Assumes retrieve_context already performed law-aware selection.
        Formatter should preserve that ordering and avoid dropping supporting
        articles aggressively at the final step.
        """
        relevant_docs = set()
        relevant_articles = set()
        relevant_doc_list = []
        relevant_article_list = []

        ordered = sorted(
            results,
            key=lambda x: x.get("combined_score", x.get("score", 0.0)),
            reverse=True,
        )

        for doc in ordered[:top_k]:
            law_title = doc.get("law_title", "")
            law_id = RAGPipeline.extract_law_id(doc.get("citation", ""), law_title)
            article = doc.get("article", "")
            
            # Đảm bảo format chuẩn: <số hiệu>|<tên văn bản>
            doc_str = f"{law_id}|{law_title}"
            article_str = f"{law_id}|{law_title}|{article}"
            
            if doc_str not in relevant_docs:
                relevant_docs.add(doc_str)
                relevant_doc_list.append(doc_str)
            if article_str not in relevant_articles:
                relevant_articles.add(article_str)
                relevant_article_list.append(article_str)
            
        return {
            "id": 1,
            "question": question,
            "answer": "Đây là câu trả lời placeholder sinh ra bởi LLM...",
            "relevant_docs": relevant_doc_list,
            "relevant_articles": relevant_article_list
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    pipeline = RAGPipeline()
    
    test_query = "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
    
    results = pipeline.retrieve_context(
        query=test_query, 
        hybrid_k=20, 
        max_hops=1, 
        final_k=10, 
        fusion_method="rrf"
    )
    
    print("\n=== RAW CANDIDATES (TOP 3) ===")
    for rank, doc in enumerate(results[:3], 1):
        print(f"Rank {rank} | Score: {doc['score']:.4f} | Source: {doc['source']}")
        print(f"Title: {doc['law_title']} | Article: {doc['article']}")
        print("=" * 60)
        
    print("\n=== Submission ===")
    submission = RAGPipeline.format_submission(test_query, results, top_k=5)
    print(json.dumps(submission, ensure_ascii=False, indent=2))
        
    pipeline.close()
