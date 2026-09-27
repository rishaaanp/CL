import re
import math
from collections import Counter

# ============================================================
# Q18: WORD SENSE DISAMBIGUATION USING NAIVE BAYES
#      Bag-of-Words + Add-1 Smoothing
# ============================================================


# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

training_data = [
    ("I love fish. The smoked bass fish was delicious.", "fish"),
    ("The bass fish swam along the line.", "fish"),
    ("He hauled in a big catch of smoked bass fish.", "fish"),
    ("The bass guitar player played a smooth jazz line.", "guitar"),
]


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

test_sentence = (
    "He loves jazz. The bass line provided the foundation "
    "for the guitar solo in the jazz piece."
)

ambiguous_word = "bass"


# ============================================================
# TOKENIZATION
# ============================================================


def tokenize(text):

    return re.findall(r"[a-z]+", text.lower())


# ============================================================
# REMOVE AMBIGUOUS WORD FROM CONTEXT
# ============================================================


def remove_ambiguous_word(text, word):

    words = tokenize(text)

    # Remove only the ambiguous word
    context_words = [w for w in words if w != word.lower()]

    return context_words


# ============================================================
# BUILD VOCABULARY
# ============================================================

vocabulary = set()

for sentence, sense in training_data:

    words = remove_ambiguous_word(sentence, ambiguous_word)

    vocabulary.update(words)


vocabulary = sorted(vocabulary)

V = len(vocabulary)


# ============================================================
# DISPLAY BASIC INFORMATION
# ============================================================

print("=" * 75)
print("NAIVE BAYES WORD SENSE DISAMBIGUATION")
print("=" * 75)

print("\nAmbiguous word:", ambiguous_word)

print("\nPossible senses:")

senses = sorted(set(sense for _, sense in training_data))

for sense in senses:

    print("-", sense)

print("\nVocabulary size:", V)


# ============================================================
# TRAIN NAIVE BAYES MODEL
# ============================================================

word_counts = {}
total_words = {}
document_counts = {}


for sense in senses:

    word_counts[sense] = Counter()
    total_words[sense] = 0
    document_counts[sense] = 0


# ------------------------------------------------------------
# COUNT CONTEXT WORDS
# ------------------------------------------------------------

for sentence, sense in training_data:

    # IMPORTANT:
    # Remove "bass" so that the classifier learns
    # only from contextual words.

    words = remove_ambiguous_word(sentence, ambiguous_word)

    document_counts[sense] += 1

    for word in words:

        word_counts[sense][word] += 1
        total_words[sense] += 1


# ============================================================
# DISPLAY TRAINING INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("TRAINING INFORMATION")
print("=" * 75)

for sense in senses:

    print("\nSense:", sense)

    print("Number of training sentences:", document_counts[sense])

    print("Total context words:", total_words[sense])

    print("Word counts:")

    print(dict(word_counts[sense]))


# ============================================================
# CLASS PRIOR
# ============================================================

total_documents = len(training_data)


def prior_probability(sense):

    return document_counts[sense] / total_documents


# ============================================================
# ADD-1 SMOOTHED WORD PROBABILITY
# ============================================================


def word_probability(word, sense):

    count = word_counts[sense][word]

    probability = (count + 1) / (total_words[sense] + V)

    return probability


# ============================================================
# CLASSIFY CONTEXT
# ============================================================


def classify_context(context_words):

    scores = {}

    for sense in senses:

        # Start with class prior
        score = math.log(prior_probability(sense))

        # Add probability of every context word
        for word in context_words:

            probability = word_probability(word, sense)

            score += math.log(probability)

        scores[sense] = score

    predicted_sense = max(scores, key=scores.get)

    return predicted_sense, scores


# ============================================================
# EXTRACT TEST CONTEXT
# ============================================================

test_context = remove_ambiguous_word(test_sentence, ambiguous_word)


# ============================================================
# DISPLAY TEST CONTEXT
# ============================================================

print("\n" + "=" * 75)
print("TEST SENTENCE")
print("=" * 75)

print(test_sentence)

print("\nAmbiguous word:", ambiguous_word)

print("\nContext used for classification:")

print(" ".join(test_context))

print(
    "\nNote: The ambiguous word",
    "'" + ambiguous_word + "'",
    "is excluded from the features.",
)


# ============================================================
# CLASSIFY
# ============================================================

predicted_sense, scores = classify_context(test_context)


# ============================================================
# DISPLAY NAIVE BAYES SCORES
# ============================================================

print("\n" + "=" * 75)
print("NAIVE BAYES SCORES")
print("=" * 75)

for sense in senses:

    print(f"{sense:10s} : " f"{scores[sense]:.6f}")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print("FINAL RESULT")
print("=" * 75)

print("The word", "'" + ambiguous_word + "'", "has the sense:", predicted_sense)

print("\nOutput:", predicted_sense)

print("=" * 75)
