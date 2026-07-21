import faiss
import numpy as np
from vectorbench_faiss.indexes.base import BaseIndex

class PQIndex(BaseIndex):
    def __init__(self, dim: int, nlist: int, nbits: int, m):
        super().__init__(dim)
        self.nlist = nlist
        self.m = m
        # 8 or 16 since dim/m should be divisible
        self.nbits = nbits
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    def train(self, vectors: np.ndarray) -> None:
        vectors = self._ensure_float32(vectors)
        self.index.train(vectors)
        self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_trained:
            raise RuntimeError("PQIndex must be trained before calling add()")

        vectors = self._ensure_float32(vectors)
        self.index.add(vectors)
        self.ntotal = self.index.ntotal

    def search(self, query_vectors: np.ndarray, k: int = 5):
        query_vectors = self._ensure_float32(query_vectors)
        distances, ids = self.index.search(query_vectors, k)
        return distances, ids

    def set_nprobe(self, nprobe: int) -> None:
        self.nprobe = nprobe
        self.index.nprobe = nprobe
    