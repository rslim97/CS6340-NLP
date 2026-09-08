import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# from nltk.corpus import stopwords  # Module for stop words that come with NLTK
from nltk.corpus import stopwords


"""
$ conda remove libstdcxx-ng --force
$ conda install -c conda-forge "libstdcxx-ng>=13" 
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
verify installation
$ strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep CXXABI
"""


torch.manual_seed(43)
# torch.seed()

# sentences = ['cat chases mice',
#              'cat catches mice',
#              'cat eats mice',
#              'mice runs into hole',
#              'cat says bad words',
#              'cat and mice are pals',
#              'cat and mice are chums',
#              'mice stores food in hole',
#              'cat stores food in house',
#              'mice sleeps in hole',
#              'cat sleeps in house',
#              'cat and mice are buddies',
#              'mice lives in hole',
#              'cat lives in house']


sentences = ['cat chases mouse',
        'cat catches mouse',
        'cat eats mouse',
        'mouse runs into hole',
        'cat says bad words',
        'cat and mouse are pals',
        'cat and mouse are chums',
        'mouse stores food in hole',
        'cat stores food in house',
        'mouse sleeps in hole',
        'cat sleeps in house',
        'cat and mouse are buddies',
        'mouse lives in hole',
        'cat lives in house']

stop_words = set(stopwords.words("english"))


def get_vocab(sentences):
    """
    sentences: type: List[sentence: string]
    """

    idx = 1  # reserve index 0 for <PAD>
    word2idx = dict()
    idx2word = dict()
    unique = set()

    for sentence in sentences:
        for word in sentence.split():
            if word not in unique and word not in stop_words:
                unique.add(word)
                word2idx[word] = idx  # Mapping word to id
                idx2word[idx] = word  # Mapping id to word
                idx += 1

    word2idx["UNK"] = 0
    idx2word[0] = "UNK"

    return word2idx, idx2word





def get_training_data(sentences, window_size, word2idx, idx2word):
    """
    sentences: List[sentence: string]
    window_size: int
    """

    def get_prev_words(idx, sentence):
        """
        return: list of indices of prev words
        """
        start_idx = max(idx - window_size // 2, 0)
        end_idx = idx - 1
        prev_words = sentence[start_idx: end_idx + 1]
        return [word2idx[word] for word in prev_words]

    def get_next_words(idx, sentence):
        """
        return list of indices of next words
        """
        start_idx = idx + 1
        end_idx = min(idx + window_size // 2, len(sentence) - 1)
        next_words = sentence[start_idx: end_idx + 1]
        return [word2idx[word] for word in next_words]

    Xs = []
    ys = []
    for i in range(len(sentences)):
        sentence = [word for word in sentences[i].split() if word not in stop_words]
        for j in range(len(sentence)):
            # if sentence[j] in stop_words:
            #     continue
            prev_words = get_prev_words(j, sentence)
            next_words = get_next_words(j, sentence)

            # PAD with zeros
            if len(prev_words) < window_size // 2:
                prev_words.extend([0]*(window_size // 2 - len(prev_words)))
            if len(next_words) < window_size // 2:
                next_words.extend([0]*(window_size // 2 - len(next_words)))
            
            print("sentence", sentence)
            print("curr_word", sentence[j])
            print("prev_words", prev_words)
            print("next_words", next_words)
            print([idx2word[idx] for idx in prev_words])
            # print(sentence[j])
            print([idx2word[idx] for idx in next_words])
            # break
            print('\n')
            X = prev_words + next_words
            y = word2idx[sentence[j]]
            Xs.append(X)
            ys.append(y)


        # break
    print("Xs", Xs)
    print("ys", ys)
    return np.asarray(Xs), np.asarray(ys)


class model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # self.fc1 = nn.Linear(embed_dim, vocab_size)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        """
        x: shape: (N, T), values \in [0, |V|], (including <PAD>)
        y: shape: (N,), values \in [0, |V|], (including <PAD>)
        """
        # print("x.shape", x.shape)
        x = self.embed_layer(x)
        # print("x.shape", x.shape)
        x = torch.mean(x, dim=1)
        # print("x.shape", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

def train_model(model, Xs, ys, idx2word):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1
    batch_size = 4
    N = Xs.shape[0]
    num_batches = N // batch_size
    for epoch in range(num_epochs):
        model.train()
        print(f"epoch: ", epoch+1)
        for i in range(num_batches):
            x = Xs[i:i+batch_size]
            y = ys[i:i+batch_size]
            print("x.shape, y", x.shape, y.shape)
            print("x, y", x, y)
            optimizer.zero_grad()
            pred = model(x)
            print("pred.shape", pred.shape)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            x_ = x.clone().detach().numpy()
            y_ = y.clone().detach().numpy()
            for b in range(batch_size):
                x_b = x_[b]
                print(f"indices: x_b: {x_b}, y_b: {y_[b]}" )
                print(f"words: x_b: {[idx2word[idx] for idx in x_b]}, y_b: {idx2word[y_[b]]}")
            print("\n")

        #     break
        # break
    # return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    word2idx, idx2word = get_vocab(sentences)
    window_size = 3
    Xs, ys = get_training_data(sentences, window_size, word2idx, idx2word)
    Xs = torch.tensor(Xs).to(device)
    ys = torch.tensor(ys).to(device)
    vocab_size = len(word2idx.keys())
    embed_dim = 2
    hidden_dim = 10
    cbow_model = model(vocab_size, embed_dim, hidden_dim).to(device)
    train_model(cbow_model, Xs, ys, idx2word)

    # print(Xs)
    # print(ys)
    # # print(len(Xs))
    # # print(len(ys))
    # print(Xs.shape)
    # print(ys.shape)

    # test_x = Xs[0].unsqueeze(0)
    # test_pred = cbow_model(test_x)
    # print(test_pred)

    word_embeddings = cbow_model.embed_layer.weight
    word_embeddings = word_embeddings.detach().cpu().numpy()
    print("word_embeddings.shape", word_embeddings.shape)

    print("len(word2idx.keys())", len(word2idx.keys()))

    plt.figure(figsize=(10, 6))
    plt.scatter([i[0] for i in word_embeddings], [i[1] for i in word_embeddings])
    for i, item in zip(idx2word.keys(), word_embeddings):
        plt.text(item[0], item[1], idx2word[i], fontdict={'fontsize': 12})
    plt.show()