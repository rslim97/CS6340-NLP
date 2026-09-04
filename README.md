#### a1
```
python sentiment_classifier.py --model PERCEPTRON --feat UNIGRAM
python sentiment_classifier.py --model LR --feat UNIGRAM
python sentiment_classifier.py --model LR --feat BIGRAM
python sentiment_classifier.py --model LR --feat BETTER
```

#### a2
1. To run SGD: 
```
python optimization.py --lr 0.1
```

2. To train Deep Averaging Network:
```
python neural_sentiment_classifier.py --batch_size 4 --num_epochs 5
```