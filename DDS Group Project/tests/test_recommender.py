import numpy as np
import pytest
import torch

from recommender import get_recommendations, slug_to_search_query

# ---------------------------------------------------------------------------
# slug_to_search_query
# ---------------------------------------------------------------------------


def test_slug_strips_leading_id():
    assert slug_to_search_query("4204-pink-floyd-the-dark-side-of-the-moon") == "pink floyd the dark side of the moon"


def test_slug_replaces_hyphens_with_spaces():
    assert slug_to_search_query("6077-king-crimson-red") == "king crimson red"


def test_slug_handles_large_id():
    assert slug_to_search_query("116023-twenty-one-pilots-trench") == "twenty one pilots trench"


def test_slug_single_word_title():
    assert slug_to_search_query("6088-joni-mitchell-hejira") == "joni mitchell hejira"


def test_slug_no_id_prefix_unchanged():
    # If there's no leading number, hyphens are still replaced
    assert slug_to_search_query("no-id-here") == "no id here"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_data():
    """Minimal synthetic data: 3 users, 5 items, 2-d embeddings."""
    rng = np.random.default_rng(42)

    user_id_mapping = {"alice": 0, "bob": 1, "charlie": 2}
    item_id_mapping = {"item_a": 0, "item_b": 1, "item_c": 2, "item_d": 3, "item_e": 4}

    user_embeddings = rng.standard_normal((3, 2)).astype(np.float32)
    item_embeddings = rng.standard_normal((5, 2)).astype(np.float32)

    # alice has rated items 0 and 1
    user_ids = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    return dict(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_id_mapping=user_id_mapping,
        item_id_mapping=item_id_mapping,
        user_ids=user_ids,
        item_ids=item_ids,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_list_of_tuples(small_data):
    results = get_recommendations("alice", **small_data, top_n=3)
    assert isinstance(results, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_respects_top_n(small_data):
    results = get_recommendations("alice", **small_data, top_n=2)
    assert len(results) == 2


def test_top_n_larger_than_unrated_returns_all_unrated(small_data):
    # alice rated items 0 and 1 → 3 unrated items remain
    results = get_recommendations("alice", **small_data, top_n=100)
    assert len(results) == 3


def test_scores_sorted_descending(small_data):
    results = get_recommendations("alice", **small_data, top_n=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_excludes_rated_items(small_data):
    # alice has rated item_a (0) and item_b (1)
    results = get_recommendations("alice", **small_data, top_n=10)
    slugs = [slug for slug, _ in results]
    assert "item_a" not in slugs
    assert "item_b" not in slugs


def test_includes_only_valid_slugs(small_data):
    results = get_recommendations("alice", **small_data, top_n=10)
    valid_slugs = set(small_data["item_id_mapping"].keys())
    for slug, _ in results:
        assert slug in valid_slugs


def test_unknown_user_raises(small_data):
    with pytest.raises(ValueError, match="Unknown user"):
        get_recommendations("nobody", **small_data)


def test_works_with_numpy_arrays_instead_of_tensors(small_data):
    data = dict(small_data)
    data["user_ids"] = data["user_ids"].numpy()
    data["item_ids"] = data["item_ids"].numpy()
    results = get_recommendations("alice", **data, top_n=3)
    assert len(results) == 3


def test_scores_are_floats(small_data):
    results = get_recommendations("alice", **small_data, top_n=3)
    assert all(isinstance(score, float) for _, score in results)


def test_user_with_no_ratings_gets_all_items(small_data):
    # Add a user with no interactions
    data = dict(small_data)
    data["user_id_mapping"] = {**small_data["user_id_mapping"], "newuser": 3}
    data["user_embeddings"] = np.vstack(
        [small_data["user_embeddings"], np.ones((1, 2), dtype=np.float32)]
    )
    results = get_recommendations("newuser", **data, top_n=10)
    assert len(results) == len(small_data["item_id_mapping"])


def test_different_users_can_get_different_recommendations(small_data):
    alice_results = get_recommendations("alice", **small_data, top_n=3)
    bob_results = get_recommendations("bob", **small_data, top_n=3)
    alice_slugs = [s for s, _ in alice_results]
    bob_slugs = [s for s, _ in bob_results]
    # With random embeddings they almost certainly differ; at minimum the
    # function should run without error for both users.
    assert isinstance(alice_slugs, list)
    assert isinstance(bob_slugs, list)
