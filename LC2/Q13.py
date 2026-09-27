import re
import math
from collections import Counter

# ============================================================
# Q13: SENTIMENT ANALYSIS USING NAIVE BAYES
# ============================================================


# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

train_data = [
    # Positive reviews
    ("I love this movie", "positive"),
    ("This movie is excellent", "positive"),
    ("The movie was amazing", "positive"),
    ("I really enjoyed this film", "positive"),
    ("This is a wonderful movie", "positive"),
    ("The acting was excellent", "positive"),
    ("The story was interesting", "positive"),
    ("I enjoyed the story", "positive"),
    ("The movie was fantastic", "positive"),
    ("I love the acting", "positive"),
    # Negative reviews
    ("I hate this movie", "negative"),
    ("This movie is terrible", "negative"),
    ("The movie was boring", "negative"),
    ("I really disliked this film", "negative"),
    ("This is a horrible movie", "negative"),
    ("The acting was terrible", "negative"),
    ("The story was boring", "negative"),
    ("I disliked the story", "negative"),
    ("The movie was awful", "negative"),
    ("I hate the acting", "negative"),
]


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

test_data = [
    ("I love this film", "positive"),
    ("The movie was amazing", "positive"),
    ("This is a wonderful film", "positive"),
    ("The acting was fantastic", "positive"),
    ("I enjoyed this movie", "positive"),
    ("I hate this film", "negative"),
    ("The movie was awful", "negative"),
    ("This is a horrible film", "negative"),
    ("The acting was boring", "negative"),
    ("I disliked this movie", "negative"),
]


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------


def tokenize(text):

    return re.findall(r"[a-z]+", text.lower())


# ------------------------------------------------------------
# BUILD NAIVE BAYES MODEL
# ------------------------------------------------------------


def train_model():

    classes = ["positive", "negative"]

    # Word counts for each class
    word_counts = {"positive": Counter(), "negative": Counter()}

    # Total words in each class
    total_words = {"positive": 0, "negative": 0}

    # Number of documents in each class
    document_counts = {"positive": 0, "negative": 0}

    # Vocabulary
    vocabulary = set()

    # Process every training document
    for text, label in train_data:

        document_counts[label] += 1

        words = tokenize(text)

        for word in words:

            word_counts[label][word] += 1
            total_words[label] += 1
            vocabulary.add(word)

    return (classes, word_counts, total_words, document_counts, vocabulary)


# ------------------------------------------------------------
# CALCULATE PRIOR PROBABILITY
# ------------------------------------------------------------


def class_prior(label, document_counts):

    total_documents = sum(document_counts.values())

    return document_counts[label] / total_documents


# ------------------------------------------------------------
# CALCULATE WORD PROBABILITY
# USING ADD-k SMOOTHING
# ------------------------------------------------------------


def word_probability(word, label, word_counts, total_words, vocabulary, k):

    word_count = word_counts[label][word]

    V = len(vocabulary)

    probability = (word_count + k) / (total_words[label] + k * V)

    return probability


# ------------------------------------------------------------
# CLASSIFY ONE DOCUMENT
# ------------------------------------------------------------


def classify(text, classes, word_counts, total_words, document_counts, vocabulary, k):

    words = tokenize(text)

    scores = {}

    for label in classes:

        # Start with log prior
        prior = class_prior(label, document_counts)

        score = math.log(prior)

        # Add log probability of every word
        for word in words:

            probability = word_probability(
                word, label, word_counts, total_words, vocabulary, k
            )

            score += math.log(probability)

        scores[label] = score

    # Select class with highest score
    predicted_class = max(scores, key=scores.get)

    return predicted_class, scores


# ------------------------------------------------------------
# EVALUATE MODEL
# ------------------------------------------------------------


def evaluate(k):

    classes, word_counts, total_words, document_counts, vocabulary = train_model()

    correct = 0

    print("\n")
    print("=" * 75)
    print(f"RESULTS FOR k = {k}")
    print("=" * 75)

    print(f"{'Text':45s} " f"{'Expected':12s} " f"{'Predicted':12s}")

    print("-" * 75)

    for text, actual_label in test_data:

        predicted_label, scores = classify(
            text, classes, word_counts, total_words, document_counts, vocabulary, k
        )

        if predicted_label == actual_label:
            correct += 1

        print(f"{text:45s} " f"{actual_label:12s} " f"{predicted_label:12s}")

    total = len(test_data)

    accuracy = (correct / total) * 100

    print("-" * 75)

    print("Correct predictions :", correct)
    print("Total test samples  :", total)
    print("Accuracy            :", round(accuracy, 2), "%")

    return accuracy


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 75)
print("NAIVE BAYES SENTIMENT CLASSIFIER")
print("=" * 75)

print("\nTraining documents:", len(train_data))
print("Test documents    :", len(test_data))


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

classes, word_counts, total_words, document_counts, vocabulary = train_model()

print("Vocabulary size    :", len(vocabulary))


# ------------------------------------------------------------
# DISPLAY CLASS INFORMATION
# ------------------------------------------------------------

print("\nClass distribution:")

for label in classes:

    print(label, ":", document_counts[label], "documents")


# ------------------------------------------------------------
# TEST THREE VALUES OF k
# ------------------------------------------------------------

k_values = [0.25, 0.75, 1.0]

results = {}


for k in k_values:

    accuracy = evaluate(k)

    results[k] = accuracy


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 75)
print("FINAL COMPARISON")
print("=" * 75)

print(f"{'k value':15s}" f"{'Accuracy':15s}")

print("-" * 30)

for k in k_values:

    print(f"{k:<15}" f"{results[k]:.2f}%")

print("=" * 75)


# ------------------------------------------------------------
# FIND BEST / TIED k VALUES
# ------------------------------------------------------------

max_accuracy = max(results.values())

best_k_values = [k for k, accuracy in results.items() if accuracy == max_accuracy]

print("\nMaximum accuracy:", round(max_accuracy, 2), "%")

if len(best_k_values) == 1:

    print("Best k value:", best_k_values[0])

else:

    print("All of the following k values achieved " "the same maximum accuracy:")

    for k in best_k_values:
        print("k =", k)
