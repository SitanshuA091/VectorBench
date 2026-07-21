import time
import numpy as np


class BaseIndex:

    def __init__(self, dim: int):
        self.dim = dim
        self.index = None          # subclasses set this to a real faiss.Index
        self.is_trained = False    # tracks whether train() has been called (if needed)
        self.ntotal = 0            # number of vectors currently stored

    def train(self, vectors: np.ndarray) -> None:
        self.is_trained = True  # nothing to train, so mark as trained immediately

    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement add()")

    def search(self, query_vectors: np.ndarray, k: int = 5):
        raise NotImplementedError("Subclasses must implement search()")

    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        import faiss
        self.index = faiss.read_index(path)
        self.is_trained = True
        self.ntotal = self.index.ntotal

    def timed_search(self, query_vectors: np.ndarray, k: int = 5):

        start = time.perf_counter()
        distances, ids = self.search(query_vectors, k=k)
        elapsed = time.perf_counter() - start
        return distances, ids, elapsed

    def _ensure_float32(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        return vectors