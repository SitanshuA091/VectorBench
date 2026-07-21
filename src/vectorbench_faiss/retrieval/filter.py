def matches_filter(metadata: dict, filter_dict: dict) -> bool:
    for field, condition in filter_dict.items():
        value = metadata.get(field)

        if isinstance(condition, dict):
            for op, target in condition.items():
                if op == "$gte" and not (value is not None and value >= target):
                    return False
                if op == "$lte" and not (value is not None and value <= target):
                    return False
                if op == "$gt" and not (value is not None and value > target):
                    return False
                if op == "$lt" and not (value is not None and value < target):
                    return False
                if op == "$eq" and value != target:
                    return False
        else:
            if value != condition:
                return False

    return True


def apply_filter(results: list, metadata_lookup: dict, filter_dict: dict) -> list:
    filtered = []
    for doc_id, score in results:
        metadata = metadata_lookup.get(doc_id, {})
        if matches_filter(metadata, filter_dict):
            filtered.append((doc_id, score))
    return filtered

# metadata filtering retreives results based on structured fields(year, source, category, etc)
# created two strategies post filter(run vector index/bm25 search) and pre filter(apply some filter restricting before search) 