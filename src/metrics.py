import numpy as np

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant)

def average_precision_at_k(recommended, relevant, k):
    ap = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
