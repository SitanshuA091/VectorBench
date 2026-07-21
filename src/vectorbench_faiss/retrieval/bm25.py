from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.doc_ids = None

    def build(self, documents: list[str], doc_ids: list):
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = doc_ids

    def search(self, query: str, k: int = 5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = scores.argsort()[::-1][:k]
        results = [(self.doc_ids[i], scores[i]) for i in top_k_idx]
        return results