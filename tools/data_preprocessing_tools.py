from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from sklearn.utils import Bunch


class TextProcessingService:
    def __init__(self, data_groups_list: Iterable['Bunch']):
        unique_words = set()
        unique_categories = set()

        for data_groups in data_groups_list:
            for text in data_groups.data:
                for word in text.split(' '):
                    unique_words.add(word.lower())

        for data_groups in data_groups_list:
            for category in data_groups.target:
                unique_categories.add(category)

        self.total_words_count = len(unique_words)
        self.categories_count = len(unique_categories)
        self.words_indices = {word: index for index, word in enumerate(unique_words)}

    def get_category_vector(self, category: int) -> np.ndarray:
        vector = np.zeros(self.categories_count, dtype=float)

        for index in range(self.categories_count):
            if category == index:
                vector[index] = 1

        return vector

    def get_text_vector(self, words: List[str]) -> np.ndarray:
        text_vector = np.zeros(self.total_words_count, dtype=float)

        for word in words:
            text_vector[self.words_indices[word.lower()]] += 1

        return text_vector

    def get_batch(
            self,
            data_groups: 'Bunch',
            index: int,
            batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start_index = index * batch_size
        end_index = start_index + batch_size

        texts = data_groups.data[start_index:end_index]
        categories = data_groups.target[start_index:end_index]

        return (
            np.array([self.get_text_vector(text.split(' ')) for text in texts]),
            np.array([self.get_category_vector(category) for category in categories])
        )
