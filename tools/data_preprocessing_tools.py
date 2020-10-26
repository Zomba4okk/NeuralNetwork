from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from sklearn.utils import Bunch


def get_category_vector(category: int, categories_count) -> np.ndarray:
    vector = np.zeros(categories_count, dtype=float)

    for index in range(categories_count):
        if category == index:
            vector[index] = 1

    return vector


def get_words_indices(words: List[str]) -> Dict[str, int]:
    return {
        unique_word: index for index, unique_word in enumerate(
            {word.lower() for word in words}
        )
    }


def get_words_vector(words: List[str], total_words_count: int) -> np.ndarray:
    words_indices = get_words_indices(words)
    words_vector = np.zeros(total_words_count, dtype=float)

    for word in words:
        words_vector[words_indices[word.lower()]] += 1

    return words_vector


def get_batch(
        data_groups: 'Bunch',
        index: int,
        batch_size: int,
        total_words_count: int,
        categories_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    start_index = index * batch_size
    end_index = start_index + batch_size

    texts = data_groups.data[start_index:end_index]
    categories = data_groups.target[start_index:end_index]

    return (
        np.array([get_words_vector(text.split(' '), total_words_count) for text in texts]),
        np.array([get_category_vector(category, categories_count) for category in categories])
    )


def get_unique_words_count(data_groups: 'Bunch') -> int:
    unique_words = set()
    for text in data_groups.data:
        for word in text.split(' '):
            unique_words.add(word)

    return len(unique_words)
