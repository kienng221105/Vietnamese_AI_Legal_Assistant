import time
import pickle
import json
from pathlib import Path

# Default paths relative to project root
DEFAULT_ADJACENCY_PATH = "data/kg/adjacency.pkl"
DEFAULT_CORPUS_PATH = "data/retrieval/retrieval_corpus.jsonl"


class GraphExpander:
    """
    Expands seed document IDs using a local adjacency list (adjacency.pkl)
    instead of requiring a Neo4j server.
    
    The adjacency list maps node_id -> [neighbor_id, ...] and is built
    from the Knowledge Graph edges (legal_graph.jsonl).
    """

    def __init__(self, adjacency_path: str = None, corpus_path: str = None):
        self.adjacency: dict[str, list[str]] = {}
        self.corpus_lookup: dict[str, dict] = {}

        adj_path = adjacency_path or DEFAULT_ADJACENCY_PATH
        corp_path = corpus_path or DEFAULT_CORPUS_PATH

        # --- Load adjacency list ---
        start = time.time()
        try:
            with open(adj_path, "rb") as f:
                self.adjacency = pickle.load(f)
            print(f"[GraphExpander] Loaded adjacency list with {len(self.adjacency)} nodes in {time.time() - start:.2f}s.", flush=True)
        except FileNotFoundError:
            print(f"[GraphExpander] WARNING: adjacency file not found at {adj_path}. Graph expansion disabled.", flush=True)
        except Exception as e:
            print(f"[GraphExpander] WARNING: Failed to load adjacency list: {e}. Graph expansion disabled.", flush=True)

        # --- Build corpus lookup (node_id -> doc metadata) ---
        start = time.time()
        try:
            with open(corp_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    nid = doc.get("node_id", "")
                    if nid:
                        self.corpus_lookup[nid] = doc
            print(f"[GraphExpander] Built corpus lookup with {len(self.corpus_lookup)} entries in {time.time() - start:.2f}s.", flush=True)
        except FileNotFoundError:
            print(f"[GraphExpander] WARNING: corpus file not found at {corp_path}. Neighbor metadata will be unavailable.", flush=True)
        except Exception as e:
            print(f"[GraphExpander] WARNING: Failed to build corpus lookup: {e}.", flush=True)

    def expand_seeds(self, seed_ids: list[str], max_hops: int = 1) -> list[dict]:
        """
        Expands the list of seed node IDs by traversing the local adjacency list
        up to `max_hops` hops. Returns neighbor documents with metadata from the
        retrieval corpus.
        """
        if not seed_ids or not self.adjacency:
            return []

        start_time = time.time()
        print(f"[GraphExpander] Expanding {len(seed_ids)} seeds with {max_hops} hops...", flush=True)

        seed_set = set(seed_ids)
        visited = set(seed_ids)
        frontier = set(seed_ids)

        # BFS expansion up to max_hops
        for hop in range(max_hops):
            next_frontier = set()
            for nid in frontier:
                neighbors = self.adjacency.get(nid, [])
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        # Collect neighbor IDs (excluding seeds)
        neighbor_ids = visited - seed_set

        # Build result documents from corpus lookup
        neighbors = []
        for nid in neighbor_ids:
            doc = self.corpus_lookup.get(nid)
            if doc and doc.get("content"):
                neighbors.append({
                    "node_id": nid,
                    "law_title": doc.get("law_title", ""),
                    "article": doc.get("article", ""),
                    "article_number": doc.get("article_number", ""),
                    "content": doc["content"],
                    "citation": doc.get("citation", ""),
                    "retrieval_text": doc.get("retrieval_text", ""),
                    "chude": doc.get("chude", ""),
                    "demuc": doc.get("demuc", ""),
                    "score": 0.0,  # Will be scored by Cross-Encoder later
                    "source": "graph_neighbor"
                })

        print(f"[GraphExpander] Expansion completed in {time.time() - start_time:.4f}s. "
              f"Found {len(neighbor_ids)} neighbor IDs, {len(neighbors)} with content.", flush=True)
        return neighbors
