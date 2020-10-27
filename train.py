from sklearn.datasets import fetch_20newsgroups

from constants import categories
from text_analyzer import TextAnalyzer


if __name__ == '__main__':
    train_news_groups = fetch_20newsgroups(subset='train', categories=categories)
    print('Train data_groups loaded')
    test_news_groups = fetch_20newsgroups(subset='test', categories=categories)
    print('Test data_groups loaded')

    text_analyzer = TextAnalyzer((train_news_groups, test_news_groups))

    text_analyzer.train_model(train_news_groups, test_data_groups=test_news_groups, path='models/test/model')