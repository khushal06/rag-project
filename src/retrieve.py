from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_retrieve(query, vector_retriever, nodes, top_k=10, final_k=5):
    # Vector search
    vector_results = vector_retriever.retrieve(query)

    # BM25 keyword search
    tokenized = [n.text.lower().split() for n in nodes]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [nodes[i] for i in top_bm25_idx]

    # Merge and deduplicate
    seen = set()
    candidates = []
    for r in vector_results:
        node = r.node
        if node.node_id not in seen:
            seen.add(node.node_id)
            candidates.append(node)
    for node in bm25_results:
        if node.node_id not in seen:
            seen.add(node.node_id)
            candidates.append(node)

    # Cross-encoder rerank
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs).tolist()

    # Sort by score only (avoid comparing node objects)
    ranked = sorted(zip(scores, range(len(candidates))), reverse=True)

    return [candidates[i] for _, i in ranked[:final_k]]