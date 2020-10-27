import numpy as np
from sklearn.datasets import fetch_20newsgroups

from constants import categories
from text_analyzer import TextAnalyzer


if __name__ == '__main__':
    train_news_groups = fetch_20newsgroups(subset='train', categories=categories)
    print('Train data_groups loaded')
    test_news_groups = fetch_20newsgroups(subset='test', categories=categories)
    print('Test data_groups loaded')

    text_analyzer = TextAnalyzer((train_news_groups, test_news_groups))
    text_analyzer.load_model('models/standard_model/model.ckpt')

    texts, correct_categories_codes = text_analyzer.service.get_batch(test_news_groups, 0, 20)

    predicted_categories = text_analyzer.get_prediction(texts)
    correct_categories = np.argmax(correct_categories_codes, 1)

    correct_predictions_count = 0
    incorrect_predictions_count = 0
    for index in range(20):
        if correct_categories[index] == predicted_categories[index]:
            correct_predictions_count += 1
        else:
            incorrect_predictions_count += 1

    print(
        f'Predicted categories: {predicted_categories}\nCorrect categories:   {correct_categories}\n'
        f'Correct predictions:   {correct_predictions_count}\nIncorrect predictions: {incorrect_predictions_count}'
    )
