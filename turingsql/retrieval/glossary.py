"""Hybrid retrieval over a small business glossary (PROJECT_PLAN.md section 8).

Ported from AdaptiveRAG (src/retrievers.py, commit 62fb23a, MIT licence, same author), scaled down to an
in-memory corpus of a few dozen entries, so FAISS and the parent/child chunking are not needed:

    BM25 + dense (e5-base-v2) -> Reciprocal Rank Fusion -> bge-reranker-base -> confidence gate

Each entry has `plain` (the business wording, from the Olist project's metrics_definitions.md) and `text` (the
same definition mapped to columns). Search and reranking use `plain`, because the cross-encoder scores
SQL-like text poorly (it ranked "Revenue: SUM(price) ..." below unrelated entries); the model receives `text`.

The gate is AdaptiveRAG's idea: when the reranker is not confident, return nothing rather than a
misleading definition. Its threshold is chosen on the Olist dev split only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

EMBED_MODEL = "intfloat/e5-base-v2"
QUERY_PREFIX, DOC_PREFIX = "query: ", "passage: "
RERANK_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60                      # as in AdaptiveRAG
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def fuse_rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank + 1) to an id's score."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass
class Hit:
    id: str
    text: str
    score: float
    term_match: bool = False


def term_in_query(term: str, query: str) -> bool:
    """The glossary term (without a parenthetical like "(AOV)") appears as whole words in the query."""
    base = re.sub(r"\s*\(.*?\)", "", term).strip().lower()
    abbrev = re.findall(r"\((\w+)\)", term)
    q = " " + " ".join(tokenize(query)) + " "
    for t in (base, *abbrev):
        words = " ".join(tokenize(t))
        # whole-word match, also accepting a plural ("orders", "customers") of the last word
        if words and (f" {words} " in q or f" {words}s " in q):
            return True
    return False


class GlossaryRetriever:
    def __init__(self, entries: list[dict], device: str = "cpu"):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.entries = entries
        self.docs = [e["text"] for e in entries]                    # what the model sees
        self.search_docs = [e.get("plain") or e["text"] for e in entries]   # what retrieval scores
        self.bm25 = BM25Okapi([tokenize(d) for d in self.search_docs])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.doc_vecs = np.asarray(self.embedder.encode([DOC_PREFIX + d for d in self.search_docs],
                                                        normalize_embeddings=True), dtype="float32")
        self.reranker = CrossEncoder(RERANK_MODEL, device=device)
        self._np = np

    @classmethod
    def from_jsonl(cls, path: Path, device: str = "cpu") -> GlossaryRetriever:
        entries = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return cls(entries, device=device)

    def candidates(self, query: str, pool: int = 10) -> list[int]:
        np = self._np
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = [int(i) for i in np.argsort(bm)[::-1][:pool]]
        qv = np.asarray(self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True), dtype="float32")[0]
        dense_rank = [int(i) for i in np.argsort(self.doc_vecs @ qv)[::-1][:pool]]
        fused = fuse_rrf([bm_rank, dense_rank])
        return sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]

    def search(self, query: str, k: int = 3, min_prob: float = 0.0) -> list[Hit]:
        """Top-k entries by reranker probability that pass the gate.

        Gate: reranker probability >= `min_prob`, OR the entry's term appears verbatim in the query. The
        reranker ranks well but is poorly calibrated on short definitions (probabilities of 0.001-0.1 for the
        right entry), so a probability floor alone either drops exact mentions ("revenue") or lets noise in.
        """
        cand = self.candidates(query)
        if not cand:
            return []
        probs = self.reranker.predict([(query, self.search_docs[i]) for i in cand])
        order = sorted(range(len(cand)), key=lambda j: float(probs[j]), reverse=True)[:k]
        hits = [Hit(self.entries[cand[j]]["id"], self.docs[cand[j]], float(probs[j]),
                    term_in_query(self.entries[cand[j]]["term"], query)) for j in order]
        return [h for h in hits if h.score >= min_prob or h.term_match]
