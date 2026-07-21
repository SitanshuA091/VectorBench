import json
import os


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_dataset(input_path: str, output_path: str, chunk_size: int = 200, overlap: int = 20):
    chunked_records = []
    doc_id = 0

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or row.get("passage") or ""

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                chunked_records.append({
                    "doc_id": doc_id,
                    "text": chunk,
                    "metadata": row.get("metadata", {}),
                })
                doc_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in chunked_records:
            f.write(json.dumps(record) + "\n")

    return output_path

