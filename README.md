# **VectorBench** <img width="40" height="40" alt="image" src="https://github.com/user-attachments/assets/6a29a923-b2d2-4f9d-90c2-e8bf6d6b84e6" />



A benchmarking playground for FAISS vector indexes. VecBench builds and compares Flat, IVF, HNSW, and PQ indexes on recall@k, latency, and memory, and layers on hybrid BM25 + vector retrieval with metadata filtering. The goal is to expose what's normally hidden inside a vector database — embedding, indexing, and retrieval as separate, inspectable pieces rather than one opaque `similarity_search()` call.

## Introduction

Generally we interact with vector search through a library call in a RAG pipeline, where a vector store handles embedding, indexing, and retrieval invisibly behind an API. VectorBench builds those pieces from scratch **(using FAISS's index implementations directly, not reimplementing the algorithms)** so one can see and measure exactly what each index type is doing, and where its tradeoffs actually show up.

## Project structure

```
VectorBench/
├── data/           # dataset download, chunking, and embedding scripts
├── indexes/        # wrapper classes around FAISS index types (IVF, HNSW, PQ, ...)
├── retrieval/       # BM25, reciprocal rank fusion, metadata filtering
├── engine.py         # VectorSearchEngine — the main user-facing interface
├── benchmark.py       # recall@k, latency, and memory measurement harness
└── configs/            # dataset, model, and index parameter configs
```

`engine.py` is the entry point most usage goes through. `data/`, `indexes/`, and `retrieval/` are the building blocks it wires together; `benchmark.py` drives engine instances repeatedly to produce comparison numbers across index types.

## Usage

1. Clone the repository
   ```
   git clone https://github.com/SitanshuA091/VectorBench
   cd VecBench
   ```

2. Install dependencies with uv
   ```
   uv sync
   ```

3. (Optional) Set a local model cache directory in `.env` at the project root, so downloaded embedding models are stored inside the project instead of your global HF cache
   ```
   HF_HOME=./.cache
   ```

4. Use the engine directly in a script or notebook
   ```python
   from engine import VectorSearchEngine

   documents = ["your first document", "your second document", "..."]
   metadatas = [{"source": "notes"}, {"source": "notes"}]

   engine = VectorSearchEngine(index_type="hnsw")
   engine.add_documents(documents, metadatas=metadatas)

   results = engine.search("a query string", k=5, mode="vector")
   for r in results:
       print(r["score"], r["text"])
   ```

5. Or run it against a real dataset via the data pipeline
   ```python
   from data.download import download_dataset
   from data.chunk import chunk_dataset
   from engine import VectorSearchEngine
   
   download_dataset(dataset_name="rajpurkar/squad", split="train", output_dir="data/raw/squad")
   chunk_dataset(input_path="data/raw/squad/raw_data.jsonl", output_path="data/raw/squad/chunked_data.jsonl")

   engine = VectorSearchEngine(index_type="ivf")
   # load chunked_data.jsonl, pass documents/metadatas into engine.add_documents(...)
   ```

6. Run the benchmark harness to compare index types
   ```python
   from benchmark import BenchmarkRunner

   engines = {"ivf": ivf_engine, "hnsw": hnsw_engine, "pq": pq_engine}
   runner = BenchmarkRunner(engines, ground_truth_key="ivf")
   results = runner.run_all(documents, queries)
   ```

## Search modes

- `mode="vector"` — pure embedding-based nearest neighbor search through the selected FAISS index
- `mode="bm25"` — pure keyword-based search, no embeddings involved
- `mode="hybrid"` — both run independently, merged via reciprocal rank fusion

Metadata filtering can be layered on top of any mode via the `filter` argument to `engine.search(...)`.

## Notes

- Embedding models are downloaded once via `sentence-transformers` and cached locally; subsequent runs load from cache with no network call.
- FAISS index classes are used directly (`faiss.IndexIVFFlat`, `faiss.IndexHNSWFlat`, `faiss.IndexIVFPQ`); this project wraps them with a consistent interface rather than reimplementing the underlying algorithms.
- Recall@k benchmarking requires a ground-truth reference index (typically an exact/uncompressed index) to compare approximate results against.
- **Upcoming** - proper comparative dashboard
