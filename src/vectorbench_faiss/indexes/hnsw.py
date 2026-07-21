import faiss
import numpy as np
from vectorbench_faiss.indexes.base import BaseIndex

class HNSWIndex(BaseIndex):
    def __init__(self, dim: int, M: int = 32, metric_type=faiss.METRIC_L2):
        super().__init__(dim)
        self.index = faiss.IndexHNSWFlat(dim, M, metric_type)

    def add(self, vectors: np.ndarray) -> None:
        vectors = self._ensure_float32(vectors)
        self.index.add(vectors)
        self.ntotal = self.index.ntotal

    def search(self, query_vectors: np.ndarray, k: int = 5):
        query_vectors = self._ensure_float32(query_vectors)
        distances, ids = self.index.search(query_vectors, k)
        return distances, ids

    def set_ef_search(self, ef_search: int):
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search