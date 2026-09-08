import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random


# Download sample twitter dataset
nltk.download("twitter_samples")

# Select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

print("Number of positive tweets: ", len(all_positive_tweets))
print("Number of negative tweets: ", len(all_negative_tweets))

print("\nThe type of all_positive_tweets is: ", type(all_positive_tweets))
print("The type of a tweet entry is: ,", type(all_negative_tweets[0]))

fig = plt.figure(figsize=(5, 5))

# Labels for the two classes
labels = "Positives", "Negative"

# Sizes for each slide
sizes = [len(all_positive_tweets), len(all_negative_tweets)]

# Declare pie chart, where the slices will be ordered and plotted CCW
plt.pie(sizes, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis("equal")

# Display the chart
plt.show()

# Looking at raw texts
# Print positive in green
print("\033[92m]" + all_positive_tweets[random.randint(0, 5000)])

# Print negative in red
print("\033[91m]" + all_negative_tweets[random.randint(0, 5000)] + "\n")

# Reset printed string color
print("\033[0;0m", end="")

# Our selected sample. Complex enough to exemplify each step
tweet = all_positive_tweets[2277]
print(tweet)

# Download the stopwords from NLTK
nltk.download("stopwords")

import re
import string

from nltk.corpus import stopwords  # Module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # Module for stemming
from nltk.tokenize import TweetTokenizer  # Module for tokenizing string

# Remove hyperlinks, twitter marks and styles
print("\033[92m" + tweet)
print("\033[94m")
# Remove old style retweet text "RT"
tweet2 = re.sub(r"^RT[\s]+", "", tweet)

# Remove hyperlinks
tweet2 = re.sub(r"https?://[^\s\n\r]+", "", tweet2)

# Remove hashtags
# Only removing the hash # sign from the word
tweet2 = re.sub(r"#", "", tweet2)

print(tweet2)

# Tokenize the string
print()
print("\033[92m" + tweet2)
print("\033[94m")

# Instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Tokenize tweets
tweet_tokens = tokenizer.tokenize(tweet2)

print()
print("Tokenized string: ")
print(tweet_tokens)

# Remove stop words and punctuations
# Import the english stop words list from NLTK
stopwords_english = stopwords.words("english")

print("Stop_words\n")
print(string.punctuation)  # Punctuations

print()
print("\033[92m")
print(tweet_tokens)
print("\033[94m")

tweets_clean = []

for word in tweet_tokens:  # Go through every word in your tokens list
    if word not in stopwords_english and word not in string.punctuation:
        tweets_clean.append(word)

print("removed stop words and punctuation: ")
print(tweets_clean)

# Stemming
print()
print("\033[92m")
print(tweets_clean)
print("\033[94m")

# Instantiate stemming class
stemmer = PorterStemmer()

# Create an empty list to store the stems
tweets_stem = []

for word in tweets_clean:
    stem_word = stemmer.stem(word)  # stemming a word
    tweets_stem.append(stem_word)  # append to the list

print("stemmed words: ")
print(tweets_stem)

from utils import process_tweet

tweet = all_positive_tweets[2277]

print()
print("\033[92m")
print(tweet)
print("\033[94m")

# Call the imported function
tweets_stem = process_tweet(tweet)  # Preprocess a given tweet

print("preprocessed tweet: ")
print(tweets_stem)
