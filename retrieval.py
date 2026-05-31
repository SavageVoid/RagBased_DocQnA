import math
from collections import Counter
import re
from typing import List, Tuple, Dict
import chromadb
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    HYBRID_RRF_K,
    BM25_K1,
    BM25_B,
    CROSS_ENCODER_MODEL,
)
from ingest import get_embedder
from sentence_transformers import CrossEncoder

_cross_encoder = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

class BM25:
    def __init__(self, corpus: List[str], k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        if self.doc_count > 0:
            self.avg_doc_len = sum(len(self._tokenize(d)) for d in corpus) / self.doc_count
        else:
            self.avg_doc_len = 0
            
        self.doc_freqs = []
        self.idf = {}
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', str(text).lower())
    
    def _build_index(self):
        if self.doc_count == 0:
            return
            
        df = Counter()
        for doc in self.corpus:
            terms = set(self._tokenize(doc))
            for term in terms:
                df[term] += 1
            self.doc_freqs.append(Counter(self._tokenize(doc)))
        
        for term, doc_freq in df.items():
            self.idf[term] = math.log(
                (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )
    
    def score(self, query: str, doc_index: int) -> float:
        if self.doc_count == 0 or self.avg_doc_len == 0:
            return 0.0
            
        query_terms = self._tokenize(query)
        doc_freqs = self.doc_freqs[doc_index]
        doc_len = sum(doc_freqs.values())
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = doc_freqs.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf[term] * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        if self.doc_count == 0:
            return []
            
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = HYBRID_RRF_K,
    top_n: int = TOP_K_RESULTS,
) -> List[Tuple[str, float]]:
    dense_ranks = {item[0]: idx + 1 for idx, item in enumerate(dense_results)}
    sparse_ranks = {item[0]: idx + 1 for idx, item in enumerate(sparse_results)}
    
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    
    rrf_scores = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_ranks:
            score += 1.0 / (k + dense_ranks[cid])
        if cid in sparse_ranks:
            score += 1.0 / (k + sparse_ranks[cid])
        rrf_scores[cid] = score
    
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]

_cached_bm25 = None
_cached_doc_count = -1
_cached_doc_ids = set()
_cached_doc_ids_list = []
_cached_corpus = []
_cached_metadatas = []

def get_bm25_index(collection) -> tuple:
    global _cached_bm25, _cached_doc_count, _cached_doc_ids, _cached_doc_ids_list, _cached_corpus, _cached_metadatas
    
    current_count = collection.count()
    if _cached_bm25 is not None and current_count == _cached_doc_count:
        all_docs = collection.get(include=["metadatas"])
        current_ids = set(all_docs["ids"])
        if current_ids == _cached_doc_ids:
            return _cached_bm25, _cached_doc_ids_list, _cached_corpus, _cached_metadatas
            
    all_docs = collection.get(include=["documents", "metadatas"])
    corpus = all_docs["documents"]
    
    _cached_bm25 = BM25(corpus)
    _cached_doc_count = len(corpus)
    _cached_doc_ids = set(all_docs["ids"])
    _cached_doc_ids_list = all_docs["ids"]
    _cached_corpus = corpus
    _cached_metadatas = all_docs["metadatas"]
    
    return _cached_bm25, _cached_doc_ids_list, _cached_corpus, _cached_metadatas

def hybrid_search(query: str, k: int = TOP_K_RESULTS) -> List[Tuple[str, str, float]]:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    try:
        bm25, chunk_ids, corpus, metadatas = get_bm25_index(collection)
    except Exception:
        return []
        
    if bm25.doc_count == 0:
        return []

    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()[0]
    
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(20, bm25.doc_count),
        include=["documents", "metadatas", "distances"],
    )
    
    dense_list = []
    if dense_results and dense_results["ids"]:
        dense_list = [
            (doc_id, 1.0 / (1.0 + dist) if dist is not None else 0.0)
            for doc_id, dist in zip(dense_results["ids"][0], dense_results["distances"][0])
        ]
    
    sparse_scores = bm25.search(query, k=min(20, bm25.doc_count))
    sparse_list = [(chunk_ids[idx], score) for idx, score in sparse_scores]
    
    fused = reciprocal_rank_fusion(dense_list, sparse_list, top_n=min(20, bm25.doc_count))
    
    id_to_data = {
        cid: (doc, meta) 
        for cid, doc, meta in zip(chunk_ids, corpus, metadatas)
    }
    
    results = []
    for cid, rrf_score in fused:
        doc, meta = id_to_data.get(cid, ("", {"source": "unknown"}))
        results.append((doc, meta.get("source", "unknown"), round(rrf_score, 4)))
    
    return results

def rerank(query: str, chunks: List[Tuple[str, str, float]], 
           top_k: int = TOP_K_RESULTS) -> List[Tuple[str, str, float]]:
    if not chunks:
        return []
        
    ce = get_cross_encoder()
    pairs = [(query, chunk_text) for chunk_text, _, _ in chunks]
    scores = ce.predict(pairs, show_progress_bar=False)
    
    scored = []
    for (chunk_text, source, _), ce_score in zip(chunks, scores):
        scored.append((chunk_text, source, round(float(ce_score), 4)))
    
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]
