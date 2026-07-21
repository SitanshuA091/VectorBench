from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


def load_chunks(chunked_path: str):
    documents = []
    doc_ids = []
    metadatas = []

    with open(chunked_path, "r") as f:
        for line in f:
            record = json.loads(line)
            documents.append(record["text"])
            doc_ids.append(record["doc_id"])
            metadatas.append(record["metadata"])

    return documents, doc_ids, metadatas


def embed_documents(documents: list, model_name: str, output_path: str):
    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    vectors = vectors.astype("float32")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, vectors)

    return vectors
