from sprig.eval import evaluate


def test_metrics_basic():
    gold = {"q1": ["d1", "d2"], "q2": ["d3"]}
    retrieved = {"q1": ["d2", "d4"], "q2": ["d5", "d3"]}
    metrics = evaluate(gold, retrieved, ks=[1, 2])
    assert metrics.hit_at_k[1] == 0.5
    assert metrics.recall_at_k[2] > 0
    assert 0 < metrics.mrr <= 1
