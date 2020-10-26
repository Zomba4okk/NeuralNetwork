from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from constants import classes_count

if TYPE_CHECKING:
    from sklearn.utils import Bunch


def get_category_vector(category: int) -> np.ndarray:
    vector = np.zeros(classes_count, dtype=int)

    for index in range(classes_count):
        if category == index:
            vector[index] = 1

    return vector


def get_words_indices(words: List[str]) -> Dict[str, int]:
    return {
        unique_word: index for index, unique_word in enumerate(
            word.lower() for word in words
        )
    }


def get_words_vector(words: List[str]) -> np.ndarray:
    words_indices = get_words_indices(words)
    words_vector = np.zeros(len(words_indices), dtype=int)

    for word in words:
        words_vector[words_indices[word.lower()]] += 1

    return words_vector


def get_batch(data_group: 'Bunch', index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    start_index = index * batch_size
    end_index = start_index + batch_size

    texts = data_group.data[start_index:end_index]
    categories = data_group.target[start_index:end_index]

    return (
        np.array(get_words_vector(text.split(' ')) for text in texts),
        np.array(get_category_vector(category) for category in categories)
    )
