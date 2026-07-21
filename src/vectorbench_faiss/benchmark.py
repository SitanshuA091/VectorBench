import time
import tracemalloc


class LatencyBenchmark:
    def __init__(self, engine):
        self.engine = engine

    def run(self, queries: list, k: int = 5, mode: str = "vector"):
        latencies = []
        for query in queries:
            start = time.perf_counter()
            self.engine.search(query, k=k, mode=mode)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        latencies.sort()
        n = len(latencies)
        avg = sum(latencies) / n
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]

        return {"avg": avg, "p95": p95, "p99": p99}


class RecallBenchmark:
    def __init__(self, engine, ground_truth_engine):
        self.engine = engine
        self.ground_truth_engine = ground_truth_engine

    def run(self, queries: list, k: int = 5, mode: str = "vector"):
        recalls = []
        for query in queries:
            true_results = self.ground_truth_engine.search(query, k=k, mode=mode)
            true_ids = set(doc_id for doc_id, _ in true_results)

            test_results = self.engine.search(query, k=k, mode=mode)
            test_ids = set(doc_id for doc_id, _ in test_results)

            overlap = len(true_ids & test_ids)
            recall = overlap / len(true_ids) if true_ids else 0.0
            recalls.append(recall)

        return sum(recalls) / len(recalls)


class MemoryBenchmark:
    def __init__(self, engine):
        self.engine = engine

    def run(self, documents: list, metadatas: list = None):
        tracemalloc.start()
        self.engine.add_documents(documents, metadatas=metadatas)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {"current_mb": current / (1024 * 1024), "peak_mb": peak / (1024 * 1024)}


class BenchmarkRunner:
    def __init__(self, engines: dict, ground_truth_key: str = "flat"):
        self.engines = engines
        self.ground_truth_key = ground_truth_key

    def run_all(self, documents, queries, metadatas=None, k=5, mode="vector"):
        results = {}
        ground_truth_engine = self.engines[self.ground_truth_key]

        for name, engine in self.engines.items():
            mem_bench = MemoryBenchmark(engine)
            memory_result = mem_bench.run(documents, metadatas=metadatas)

            latency_bench = LatencyBenchmark(engine)
            latency_result = latency_bench.run(queries, k=k, mode=mode)

            if name == self.ground_truth_key:
                recall_result = 1.0
            else:
                recall_bench = RecallBenchmark(engine, ground_truth_engine)
                recall_result = recall_bench.run(queries, k=k, mode=mode)

            results[name] = {
                "memory": memory_result,
                "latency": latency_result,
                "recall": recall_result,
            }

        return results