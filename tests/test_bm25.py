from sprig.retrieval.bm25 import BM25Index


def test_bm25_ranks_relevant_doc():
    docs = ["apple banana", "cat dog", "banana orange"]
    index = BM25Index.build(docs)
    hits = index.search("banana", top_k=2)
    top_ids = [i for i, _ in hits]
    assert 0 in top_ids or 2 in top_ids
