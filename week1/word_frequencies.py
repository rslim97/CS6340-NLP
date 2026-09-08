import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np


nltk.download("twitter_samples")
nltk.download("stopwords")

from utils import process_tweet, build_freqs

all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")


# Concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

print("Number of tweets: ", len(tweets))

labels = np.append(
    np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets))
)

freqs = build_freqs(tweets, labels)

# Check data type
print(f"type(freqs) = {type(freqs)}")

# Check length of the dictionary
print(f"len(freqs) = {len(freqs)}")

print(freqs)

# Select some words to appear in the report. we will assume that each word is unique (i.e. no duplicates)
keys = [
    "happi",
    "merri",
    "nice",
    "good",
    "bad",
    "sad",
    "mad",
    "best",
    "pretti",
    "❤",
    ":)",
    ":(",
    "😒",
    "😬",
    "😄",
    "😍",
    "♛",
    "song",
    "idea",
    "power",
    "play",
    "magnific",
]

# list representing our table of word counts.
# each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]
data = []

# loop through our selected words
for word in keys:

    # initialize positive and negative counts
    pos = 0
    neg = 0

    # retrieve number of positive counts
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]

    # retrieve number of negative counts
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]

    # append the word counts to the table
    data.append([word, pos, neg])

data

fig, ax = plt.subplots(figsize=(8, 8))

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data])

# Plot a dot for each pair of words
ax.scatter(x, y)

# assign axis labels
plt.xlabel("Log Positive count")
plt.ylabel("Log Negative count")

# Add the word as the label at the same position as you added the points just before
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0, 9], [0, 9], color="red")  # Plot the red line that divides the 2 areas.
plt.show()
