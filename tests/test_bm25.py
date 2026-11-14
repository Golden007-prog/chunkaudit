from chunkaudit import BM25Index

TEXTS = [
    "the moon pulls the oceans and causes tides",
    "bread rises because yeast makes carbon dioxide",
    "the moon orbits the earth every month",
    "bees dance to share where the flowers are",
]


def test_best_match_comes_first():
    index = BM25Index(TEXTS)
    assert index.search("what causes tides", 2)[0] == 0
    assert index.search("yeast bread", 1) == [1]


def test_only_matching_texts_are_returned():
    index = BM25Index(TEXTS)
    assert index.search("moon", 10) == [2, 0] or index.search("moon", 10) == [0, 2]
    assert set(index.search("moon", 10)) == {0, 2}
    assert index.search("submarine", 5) == []
    assert index.search("", 5) == []


def test_k_limits_the_result_and_ties_break_by_position():
    index = BM25Index(["alpha beta", "alpha beta", "alpha beta", "gamma"])
    assert index.search("alpha", 2) == [0, 1]
    assert index.search("alpha", 10) == [0, 1, 2]


def test_rare_terms_outweigh_common_ones():
    index = BM25Index(["common word here", "common word there", "common rare word"])
    assert index.search("common rare", 1) == [2]


def test_empty_index():
    index = BM25Index([])
    assert index.search("anything", 3) == []
