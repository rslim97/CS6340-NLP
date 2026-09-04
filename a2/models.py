# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from utils import *


random.seed(100)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


def fetch_batch(data, batch_size, indexer):
    """
    data: type: List[SentimentExample]
    SentimentExample:
        - words: type: List[string]
        - label: type: int
    """
    # Optional: First we sort so that the sentences are arranged from shorter to
    # longer sentences, and sentences of similar sizes are together
    # data_sorted = sorted(data, key=lambda example: len(example.words))
    batch = []
    for start in range(0, len(data), batch_size):
        # batch_data = data_sorted[start:start + batch_size]
        batch_data = data[start : start + batch_size]
        max_len = max([len(example.words) for example in batch_data])
        # List[SentimentExample with "PAD"s in the word attribute]
        padded_sentences = [
            example.words + ["PAD"] * (max_len - len(example.words))
            for example in batch_data
        ]
        # Convert words to their indices
        sentences_as_indices = []
        for sentence in padded_sentences:
            word_indices = []
            for word in sentence:
                if indexer.contains(word):
                    word_indices.append(indexer.index_of(word))
                else:
                    word_indices.append(indexer.index_of("UNK"))
            sentences_as_indices.append(word_indices)

        batch_labels = [example.label for example in batch_data]
        batch.append(
            (
                torch.tensor(sentences_as_indices, dtype=torch.long),
                torch.tensor(batch_labels, dtype=torch.long),
            )
        )

    return batch


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, num_classes, hidden_dim, indexer: Indexer):
        self.indexer = indexer
        self.vocab_size = len(self.indexer)
        self.dan = self.deep_averaging_network(num_classes, self.vocab_size, hidden_dim)

    class deep_averaging_network(nn.Module):
        def __init__(self, num_classes, vocab_size, hidden_dim=300):
            super().__init__()
            self.num_classes = num_classes
            self.vocab_size = vocab_size
            self.embed_dim = 300
            self.hidden_dim = hidden_dim
            self.embeddings = nn.Embedding(
                self.vocab_size + 1, self.embed_dim
            )  # +1 for UNK token
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
            self.log_probs = nn.LogSoftmax(dim=1)

        def forward(self, x):
            embeddings = self.embeddings(x)
            avg = torch.mean(embeddings, dim=1)
            logits = self.classifier(avg)
            out = self.log_probs(logits)
            return out

    def predict(self, ex_words, has_typos):
        # First, handle unknown words by mapping them to "UNK"
        unk_index = self.indexer.index_of("UNK")
        word_indices = []
        for word in ex_words:
            if self.indexer.contains(word):
                word_indices.append(self.indexer.index_of(word))
            else:
                word_indices.append(unk_index)
        word_indices = torch.tensor(word_indices)
        log_probs = self.dan(word_indices.unsqueeze(0))
        pred = torch.argmax(log_probs, dim=1).item()
        return pred


def train_deep_averaging_network(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
    train_model_for_typo_setting: bool,
) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    indexer = Indexer()
    indexer.add_and_get_index("UNK")
    # print("train_exs", train_exs)
    for ex in train_exs:
        print("ex", ex)
        for word in ex.words:
            # print(word)
            indexer.add_and_get_index(word)

    num_classes = 2
    hidden_dim = 300
    classifier = NeuralSentimentClassifier(num_classes, hidden_dim, indexer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.dan.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1e-4)
    criterion = nn.NLLLoss()

    # train_exs: type: List[SentimentExample]
    random.shuffle(train_exs)

    # batches: (sentences_in_indices: shape: [batch_size, len], labels: shape: [batch_size])
    batches = fetch_batch(train_exs, args.batch_size, indexer)

    for epoch in range(args.num_epochs):
        classifier.dan.train()
        train_loss = 0.0
        for data, target in batches:
            print("data.shape, target.shape", data.shape, target.shape)
            print("data, target", data, target)
            optimizer.zero_grad()
            pred = classifier.dan(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(batches)}")

    return classifier
