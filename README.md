# Perceptron-Classifier-Vanilla-Averaged
## Perceptron classifiers (vanilla and averaged) to identify hotel reviews as either truthful or deceptive, and either positive or negative

**DataSet contains following files:**

1. One file train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). The first 3 tokens in each line are:

- a unique 7-character alphanumeric identifier
- a label True or Fake
- a label Pos or Neg

These are followed by the text of the review.

2. One file dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).

3. One file dev-key.txt with the corresponding labels for the development data, to serve as an answer key.

**Project is divided in two phases:**

- Use perceplearn.py to learn the paramaters for vanilla and averaged model for truthful or deceptive, and either positive or negative classification.
- Use percepclassify.py to classify the testset reviews.

**The learning program will be invoked in the following way:**

```
python perceplearn.py /path/to/input
```

**The classification program will be invoked in the following way:**
```
python percepclassify.py /path/to/model /path/to/input
```

**Accuracy is calculated using below code:**

```
python accuracy.py
```
