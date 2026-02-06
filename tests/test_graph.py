from sprig.retrieval.graph import build_bipartite_graph, build_term_bipartite_graph, ppr_search


def test_graph_ppr_hits_doc():
    doc_ids = ["doc1", "doc2"]
    docs = ["Alice went home", "Bob stayed"]
    index = build_bipartite_graph(doc_ids, docs, ner_mode="regex")
    hits = ppr_search(index, "Alice", top_k=1, ner_mode="regex")
    assert hits[0][0] == 0


def test_graph_ppr_seed_docs():
    doc_ids = ["doc1", "doc2"]
    docs = ["No entities here", "Still no entities"]
    index = build_bipartite_graph(doc_ids, docs, ner_mode="regex")
    hits = ppr_search(index, "Unknown", top_k=1, ner_mode="regex", seed_docs=[(1, 1.0)])
    assert hits[0][0] == 1


def test_term_graph_ppr_terms():
    doc_ids = ["doc1", "doc2"]
    docs = ["alpha beta gamma", "delta epsilon"]
    index = build_term_bipartite_graph(doc_ids, docs, min_df=1, max_df_ratio=1.0)
    hits = ppr_search(index, "alpha", top_k=1, seed_terms=["alpha"])
    assert hits[0][0] == 0
