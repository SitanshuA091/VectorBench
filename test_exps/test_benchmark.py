from datasets import load_dataset
from engine import VectorSearchEngine
from benchmark import BenchmarkRunner

def load_ms_marco(split: str = "train", n_queries: int = 1000, version: str = "v1.1"):
    dataset = load_dataset("microsoft/ms_marco", version, split=split)
    dataset = dataset.shuffle(seed=42).select(range(n_queries))

    seen_passages = {}
    documents = []
    metadatas = []
    queries = []
    relevant_doc_ids = []

    for row in dataset:
        query = row["query"]
        passages = row["passages"]["passage_text"]
        is_selected = row["passages"]["is_selected"]

        query_relevant_ids = []

        for passage_text, selected in zip(passages, is_selected):
            if passage_text not in seen_passages:
                doc_id = len(documents)
                seen_passages[passage_text] = doc_id
                documents.append(passage_text)
                metadatas.append({"source": "ms_marco"})
            else:
                doc_id = seen_passages[passage_text]

            if selected == 1:
                query_relevant_ids.append(doc_id)

        if query_relevant_ids:
            queries.append(query)
            relevant_doc_ids.append(query_relevant_ids)

    return documents, metadatas, queries, relevant_doc_ids


documents, metadatas, queries, relevant_doc_ids = load_ms_marco(n_queries=500)
sample_queries = queries[:50]

print(f"Loaded {len(documents)} unique passages, benchmarking on {len(sample_queries)} queries")

ivf_engine = VectorSearchEngine(index_type="ivf", nlist=10)
hnsw_engine = VectorSearchEngine(index_type="hnsw")
pq_engine = VectorSearchEngine(index_type="pq", nlist=10, m=8, nbits=8)

engines = {
    "ivf": ivf_engine,
    "hnsw": hnsw_engine,
    "pq": pq_engine,
}

runner = BenchmarkRunner(engines, ground_truth_key="hnsw")
results = runner.run_all(documents, sample_queries, metadatas=metadatas, k=5, mode="vector")

for name, metrics in results.items():
    print(f"\n{name}")
    print(f"  recall@5: {metrics['recall']:.3f}")
    print(f"  latency avg: {metrics['latency']['avg']*1000:.2f} ms, p95: {metrics['latency']['p95']*1000:.2f} ms")
    print(f"  memory peak: {metrics['memory']['peak_mb']:.2f} MB")