from datasets import load_dataset
from engine import VectorSearchEngine


dataset = load_dataset("rajpurkar/squad", split="train[:200]")

seen = set()
documents = []
metadatas = []

for row in dataset:
    context = row["context"]
    if context in seen:
        continue
    seen.add(context)
    documents.append(context)
    metadatas.append({"title": row["title"]})

print(f"Loaded {len(documents)} unique documents")

engine = VectorSearchEngine(index_type="hnsw")
engine.add_documents(documents, metadatas=metadatas)

query = "What is the speed of light?"
results = engine.search(query, k=5, mode="vector")

print(f"\nQuery: {query}\n")
for r in results:
    print(f"score={r['score']:.4f} title={r['metadata']['title']}")
    print(r["text"][:200])
    print("-" * 40)