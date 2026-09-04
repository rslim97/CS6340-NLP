# models.py

from sentiment_data import *
from utils import *

from collections import Counter
from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def vocab_size(self):
        return len(self.indexer)

    def extract_features(
        self, sentence: SentimentExample, add_to_indexer: bool = False
    ):
        features = Counter()
        for word in sentence:
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
            if index != -1:
                features[index] += 1

        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def vocab_size(self):
        return len(self.indexer)

    def extract_features(self, sentence: SentimentExample, add_to_indexer=False):
        features = Counter()
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
            if index != -1:
                features[index] += 1

        return features


# Implementing BetterFeatureExtractor
# bigram with clipping rare words and removing stop words
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer, idf_values: dict = None):
        self.indexer = indexer
        self.idf_values = idf_values or defaultdict(lambda: 1.0)

    def get_indexer(self):
        return self.indexer

    def vocab_size(self):
        return len(self.indexer)

    def extract_features(self, sentence, add_to_indexer=False):
        features = Counter()
        for i in range(len(sentence) - 1):
            bigram = (sentence[i].lower(), sentence[i + 1].lower())
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
            if index != -1:
                features[index] += 1

        # Apply IDF weighting
        for index in features:
            word = self.indexer.get_object(index)
            features[index] *= self.idf_values.get(word, 1.0)

        return features


def calculate_idf(train_exs: List[SentimentExample]) -> dict:
    """
    Calculate the IDF values for the terms in the training data.
    """

    num_docs = len(train_exs)
    # defaultdict are like dicts but with default value, so we can
    # access key-value pairs that don't exist yet, so we don't have
    # to always initialize the key-value pairs if we don't know what
    # the value shall be because we can just use a default value.
    doc_freq = defaultdict(int)

    # Build document frequency for each word in the training examples
    for ex in train_exs:
        # Get unique words in a document
        unique_words = set(word.lower() for word in ex.words)
        for word in unique_words:
            doc_freq[word] += 1

    # Calculate the IDF values for each word
    idf_values = {}
    for word, freq in doc_freq.items():
        idf_values[word] = np.log(
            num_docs / (1 + freq)
        )  # Apply IDF formula: log(N / (1 + DF))

    return idf_values


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence):
        features = self.feat_extractor.extract_features(sentence, False)
        score = sum(self.weights[index] * count for index, count in features.items())
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        # Compute the induced local field, score = w.T * x
        score = sum(self.weights[index] * count for index, count in features.items())
        e = np.exp(score)
        prob = 1 / (1 + np.power(e, -1))
        return 1 if prob >= 0.5 else 0


def train_perceptron(
    train_exs: List[SentimentExample], feat_extractor: FeatureExtractor
) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    num_features = feat_extractor.vocab_size()
    weights = np.zeros(num_features)
    initial_lr = 1e-2
    num_epochs = 24
    lr_decay = 1e-4
    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)
        lr = initial_lr - (epoch * lr_decay)
        lr = max(lr, 1e-5)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[index] * count for index, count in features.items())
            prediction = 1 if score >= 0 else 0
            if prediction != ex.label:
                for index, count in features.items():
                    gradient = (prediction - ex.label) * count  # (y_hat - y) * x
                    weights[index] -= lr * gradient

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(
    train_exs: List[SentimentExample], feat_extractor: FeatureExtractor
) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    num_features = feat_extractor.vocab_size()
    weights = np.zeros(num_features)
    initial_lr = 2e-2
    num_epochs = 66
    lr_decay = 1e-4

    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)
        lr = initial_lr - (epoch * lr_decay)
        lr = max(lr, 1e-6)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            # Compute the induced local field, score = w.T * x
            score = sum(weights[index] * count for index, count in features.items())
            e = np.exp(score)
            prob = 1 / (1 + np.power(e, -1))
            # weights -= lr * gradient
            for index, count in features.items():
                gradient = (prob - ex.label) * count  # (y_hat - y) * x
                weights[index] -= lr * gradient

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(
    args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]
) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # Indexer is used to perform white space and puctuation tokenization.
    # Tokenization is the process of splitting a text into words or subwords,
    # which are then converted to ids through a look-up table.
    indexer = Indexer()
    stop_words = set(stopwords.words("english"))
    punctuation = (
        ",",
        ".",
        "...",
        "?",
        "'",
        "''",
        "!",
        ":",
        ";",
        "|",
        "&",
        "#",
        "$",
        "%",
    )
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        cleaned_words = set()
        for example in train_exs:
            for word in example.words:
                if word.lower() not in stop_words and word.lower() not in punctuation:
                    cleaned_words.add(word.lower())
        for word in cleaned_words:
            # print(word)
            indexer.add_and_get_index(word)
        feat_extractor = UnigramFeatureExtractor(indexer)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        for example in train_exs:
            for i in range(len(example.words) - 1):
                bigram = (example.words[i].lower(), example.words[i + 1].lower())
                indexer.add_and_get_index(bigram)
        feat_extractor = BigramFeatureExtractor(indexer)
    elif args.feats == "BETTER":
        # Get IDF values
        idf_values = calculate_idf(train_exs)
        # print("idf_values", idf_values)
        for example in train_exs:
            for i in range(len(example.words) - 1):
                bigram = (example.words[i].lower(), example.words[i + 1].lower())
                indexer.add_and_get_index(bigram)
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(indexer, idf_values)
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system"
        )

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception(
            "Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system"
        )
    return model


"""
$ python sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM
Namespace(model='PERCEPTRON', feats='UNIGRAM', train_path='data/train.txt', dev_path='data/dev.txt', blind_test_path='data/test-blind.txt', test_output_path='test-blind.output.txt', run_on_test=True)
6920 / 872 / 1821 train/dev/test examples
=====Train Accuracy=====
Accuracy: 6835 / 6920 = 0.987717
Precision (fraction of predicted positives that are correct): 3576 / 3627 = 0.985939; Recall (fraction of true positives predicted correctly): 3576 / 3610 = 0.990582; F1 (harmonic mean of precision and recall): 0.988255
=====Dev Accuracy=====
Accuracy: 658 / 872 = 0.754587
Precision (fraction of predicted positives that are correct): 351 / 472 = 0.743644; Recall (fraction of true positives predicted correctly): 351 / 444 = 0.790541; F1 (harmonic mean of precision and recall): 0.766376
Time for training and evaluation: 0.67 seconds

$ python sentiment_classifier.py --model LR --feats UNIGRAM
Namespace(model='LR', feats='UNIGRAM', train_path='data/train.txt', dev_path='data/dev.txt', blind_test_path='data/test-blind.txt', test_output_path='test-blind.output.txt', run_on_test=True)
6920 / 872 / 1821 train/dev/test examples
=====Train Accuracy=====
Accuracy: 6701 / 6920 = 0.968353
Precision (fraction of predicted positives that are correct): 3508 / 3625 = 0.967724; Recall (fraction of true positives predicted correctly): 3508 / 3610 = 0.971745; F1 (harmonic mean of precision and recall): 0.969730
=====Dev Accuracy=====
Accuracy: 669 / 872 = 0.767202
Precision (fraction of predicted positives that are correct): 354 / 467 = 0.758030; Recall (fraction of true positives predicted correctly): 354 / 444 = 0.797297; F1 (harmonic mean of precision and recall): 0.777168
Time for training and evaluation: 2.75 seconds

$ python sentiment_classifier.py --model LR --feats BIGRAM
Namespace(model='LR', feats='BIGRAM', train_path='data/train.txt', dev_path='data/dev.txt', blind_test_path='data/test-blind.txt', test_output_path='test-blind.output.txt', run_on_test=True)
6920 / 872 / 1821 train/dev/test examples
=====Train Accuracy=====
Accuracy: 6888 / 6920 = 0.995376
Precision (fraction of predicted positives that are correct): 3607 / 3636 = 0.992024; Recall (fraction of true positives predicted correctly): 3607 / 3610 = 0.999169; F1 (harmonic mean of precision and recall): 0.995584
=====Dev Accuracy=====
Accuracy: 626 / 872 = 0.717890
Precision (fraction of predicted positives that are correct): 344 / 490 = 0.702041; Recall (fraction of true positives predicted correctly): 344 / 444 = 0.774775; F1 (harmonic mean of precision and recall): 0.736617
Time for training and evaluation: 4.69 seconds

python sentiment_classifier.py --model LR --feats BETTER
Namespace(model='LR', feats='BETTER', train_path='data/train.txt', dev_path='data/dev.txt', blind_test_path='data/test-blind.txt', test_output_path='test-blind.output.txt', run_on_test=True)
6920 / 872 / 1821 train/dev/test examples
=====Train Accuracy=====
Accuracy: 6918 / 6920 = 0.999711
Precision (fraction of predicted positives that are correct): 3608 / 3608 = 1.000000; Recall (fraction of true positives predicted correctly): 3608 / 3610 = 0.999446; F1 (harmonic mean of precision and recall): 0.999723
=====Dev Accuracy=====
Accuracy: 637 / 872 = 0.730505
Precision (fraction of predicted positives that are correct): 348 / 487 = 0.714579; Recall (fraction of true positives predicted correctly): 348 / 444 = 0.783784; F1 (harmonic mean of precision and recall): 0.747583
Time for training and evaluation: 7.21 seconds

"""
