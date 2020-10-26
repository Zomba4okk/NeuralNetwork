import numpy as np
from sklearn.datasets import fetch_20newsgroups

from constants import (
    batch_size,
    categories,
    classes_count,
    display_step,
    hidden_1_size,
    hidden_2_size,
    learning_rate,
    training_epochs,
)
from text_analyzer import TextAnalyzer
from tools.data_preprocessing_tools import get_batch, get_unique_words_count


if __name__ == '__main__':
    ### Train model ###

    train_news_groups = fetch_20newsgroups(subset='train', categories=categories)
    print('Train data_groups loaded')
    test_news_groups = fetch_20newsgroups(subset='test', categories=categories)
    print('Test data_groups loaded')

    total_words_count = get_unique_words_count(train_news_groups) + get_unique_words_count(test_news_groups)

    text_analyzer = TextAnalyzer(total_words_count, classes_count, hidden_1_size, hidden_2_size)

    text_analyzer.train_model(
        train_news_groups,
        learning_rate,
        training_epochs,
        batch_size,
        display_step,
        test_groups=test_news_groups,
        path='C:/Users/Antic/Desktop/a/NeuralNetwork/model.ckpt'
    )

    ### Test model ###
    # texts, correct_categories_codes = get_batch(test_news_groups, 0, 10, total_words_count, classes_count)
    #
    # predicted_categories = text_analyzer.get_prediction(texts)
    #
    # print(
    #     f'Predicted categories: {predicted_categories}\nCorrect categories:   {np.argmax(correct_categories_codes, 1)}'
    # )
