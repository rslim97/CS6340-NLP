import re
from nltk import ngrams

# Extracting n-grams

def n_gram_extactor(sentence, n):
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    for i in range(len(tokens) - n + 1):
        print(tokens[i:i + n])


if __name__ == '__main__':
    sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance byh Indian Army! Awesome airshow! @india_official \
#India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)'

    # n_gram_extactor(sentence, 1)
    # n_gram_extactor(sentence, 2)

    # tokens = list(ngrams('The cute little boy is playing with the kitten.'.split(), 2))
    tokens = list(ngrams('The cute little boy is playing with the kitten.'.split(), 3))
    print(tokens)