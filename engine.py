from sentence_transformers import SentenceTransformer

from indexes.ivf import IVFIndex
from indexes.hnsw import HNSWIndex
from indexes.pq import PQIndex

from retrieval.bm25 import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.filter import apply_filter


INDEX_REGISTRY = {
    "ivf": IVFIndex,
    "hnsw": HNSWIndex,
    "pq": PQIndex,
}


class VectorSearchEngine:
    def __init__(self, index_type: str, embedding_model: str = "all-MiniLM-L6-v2", **index_kwargs):
        if index_type not in INDEX_REGISTRY:
            raise ValueError(f"Unknown index_type '{index_type}', choose from {list(INDEX_REGISTRY.keys())}")

        self.index_type = index_type
        self.embedder = SentenceTransformer(embedding_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        index_class = INDEX_REGISTRY[index_type]
        self.index = index_class(dim=self.dim, **index_kwargs)

        self.bm25 = BM25Retriever()

        self.doc_texts = {}
        self.doc_metadatas = {}
        self._next_id = 0

    def add_documents(self, documents: list, metadatas: list = None):
        if metadatas is None:
            metadatas = [{} for _ in documents]

        doc_ids = list(range(self._next_id, self._next_id + len(documents)))
        self._next_id += len(documents)

        for doc_id, text, metadata in zip(doc_ids, documents, metadatas):
            self.doc_texts[doc_id] = text
            self.doc_metadatas[doc_id] = metadata

        vectors = self.embedder.encode(documents, convert_to_numpy=True).astype("float32")

        self.index.train(vectors)
        self.index.add(vectors)

        self.bm25.build(documents, doc_ids)

    def search(self, query: str, k: int = 5, mode: str = "vector", filter: dict = None):
        if mode == "vector":
            results = self._vector_search(query, k)
        elif mode == "bm25":
            results = self.bm25.search(query, k=k)
        elif mode == "hybrid":
            vector_results = self._vector_search(query, k=k)
            bm25_results = self.bm25.search(query, k=k)

            vector_ids = [doc_id for doc_id, _ in vector_results]
            bm25_ids = [doc_id for doc_id, _ in bm25_results]

            results = reciprocal_rank_fusion(vector_ids, bm25_ids, top_k=k)
        else:
            raise ValueError(f"Unknown mode '{mode}', choose from 'vector', 'bm25', 'hybrid'")

        if filter is not None:
            results = apply_filter(results, self.doc_metadatas, filter)

        return self._format_results(results)

    def _vector_search(self, query: str, k: int):
        query_vector = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, ids = self.index.search(query_vector, k=k)

        results = [(int(doc_id), float(distance)) for doc_id, distance in zip(ids[0], distances[0])]
        return results

    def _format_results(self, results: list):
        formatted = []
        for doc_id, score in results:
            formatted.append({
                "doc_id": doc_id,
                "text": self.doc_texts.get(doc_id),
                "metadata": self.doc_metadatas.get(doc_id),
                "score": score,
            })
        return formatted