import re

import numpy as np
import torch


def slug_to_search_query(slug: str) -> str:
    """Convert an album slug to a Spotify-friendly search string.

    Strips the leading numeric ID and replaces hyphens with spaces.

    Example:
        "4204-pink-floyd-the-dark-side-of-the-moon" -> "pink floyd the dark side of the moon"
    """
    without_id = re.sub(r"^\d+-", "", slug)
    return without_id.replace("-", " ")


def get_recommendations(
    username: str,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_id_mapping: dict[str, int],
    item_id_mapping: dict[str, int],
    user_ids: "torch.Tensor | np.ndarray",
    item_ids: "torch.Tensor | np.ndarray",
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Return top-n recommended (slug, score) pairs for a user, excluding already-rated items.

    Args:
        username: The username to generate recommendations for.
        user_embeddings: Array of shape (num_users, embedding_dim).
        item_embeddings: Array of shape (num_items, embedding_dim).
        user_id_mapping: Maps username -> integer user index.
        item_id_mapping: Maps slug -> integer item index.
        user_ids: 1-D array/tensor of user indices in the interaction dataset.
        item_ids: 1-D array/tensor of item indices in the interaction dataset (aligned with user_ids).
        top_n: Number of recommendations to return.

    Returns:
        List of (slug, score) tuples sorted by score descending.

    Raises:
        ValueError: If username is not found in user_id_mapping.
    """
    if username not in user_id_mapping:
        raise ValueError(f"Unknown user: {username!r}")

    user_id = user_id_mapping[username]
    user_emb = user_embeddings[user_id]

    if isinstance(user_ids, torch.Tensor):
        user_ids = user_ids.cpu().numpy()
    if isinstance(item_ids, torch.Tensor):
        item_ids = item_ids.cpu().numpy()

    rated_item_ids = item_ids[user_ids == user_id]
    all_item_ids = np.arange(len(item_id_mapping))
    unrated_item_ids = np.setdiff1d(all_item_ids, rated_item_ids)

    scores = item_embeddings[unrated_item_ids] @ user_emb
    top_indices = np.argsort(scores)[-top_n:][::-1]

    id_to_slug = {idx: slug for slug, idx in item_id_mapping.items()}
    return [(id_to_slug[unrated_item_ids[i]], float(scores[i])) for i in top_indices]
