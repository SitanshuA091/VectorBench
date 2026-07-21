from datasets import load_dataset

from vectorbench_faiss.engine import VectorSearchEngine


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


documents, metadatas, queries, relevant_doc_ids = load_ms_marco(n_queries=1000)

print(f"Loaded {len(documents)} unique passages")
print(f"Loaded {len(queries)} queries with ground truth")

engine = VectorSearchEngine(index_type="hnsw")
engine.add_documents(documents, metadatas=metadatas)

query_idx = 0
query = queries[query_idx]
true_ids = set(relevant_doc_ids[query_idx])

results = engine.search(query, k=5, mode="vector")

print(f"\nQuery: {query}")
print(f"Ground truth doc_ids: {true_ids}\n")

for r in results:
    hit = "HIT" if r["doc_id"] in true_ids else ""
    print(f"score={r['score']:.4f} doc_id={r['doc_id']} {hit}")
    print(r["text"][:200])
    print("-" * 40)