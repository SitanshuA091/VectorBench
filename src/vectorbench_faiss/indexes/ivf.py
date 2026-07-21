import faiss
import numpy as np
from vectorbench_faiss.indexes.base import BaseIndex

class IVFIndex(BaseIndex):
    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 8):
        super().__init__(dim)
        self.nlist = nlist
        self.nprobe = nprobe

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        # nprobe will be a runtime search parameter, not whilst building
        # set it now so search() picks it up automatically.
        self.index.nprobe = self.nprobe

    def train(self, vectors: np.ndarray) -> None:
        vectors = self._ensure_float32(vectors)
        self.index.train(vectors)
        self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_trained:
            raise RuntimeError("IVFIndex must be trained before calling add()")

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