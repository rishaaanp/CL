import re
from collections import Counter

# ============================================================
# Q15: BIGRAMS AND SENTENCE PROBABILITY
# ============================================================


# ------------------------------------------------------------
# TOKENIZATION FUNCTION
# ------------------------------------------------------------


def tokenize(text):

    return re.findall(r"[a-z]+", text.lower())


# ------------------------------------------------------------
# READ CORPUS
# ------------------------------------------------------------

with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()


# ------------------------------------------------------------
# GENERATE BIGRAMS
# ------------------------------------------------------------

bigram_counts = Counter()
unigram_counts = Counter()

# Process each sentence separately
sentences = [line.strip() for line in corpus.splitlines() if line.strip()]


for sentence in sentences:

    words = tokenize(sentence)

    # Add sentence boundary markers
    words = ["<s>"] + words + ["</s>"]

    # Count unigrams
    for word in words[:-1]:
        unigram_counts[word] += 1

    # Count bigrams
    for i in range(len(words) - 1):

        bigram = (words[i], words[i + 1])

        bigram_counts[bigram] += 1


# ------------------------------------------------------------
# DISPLAY BIGRAM COUNTS
# ------------------------------------------------------------

print("=" * 70)
print("BIGRAM FREQUENCY TABLE")
print("=" * 70)

print(f"{'Bigram':35s}" f"{'Count':10s}")

print("-" * 70)

for (word1, word2), count in sorted(bigram_counts.items()):

    bigram_text = word1 + " " + word2

    print(f"{bigram_text:35s}" f"{count:<10d}")


# ------------------------------------------------------------
# CALCULATE BIGRAM PROBABILITY
# ------------------------------------------------------------


def bigram_probability(word1, word2):

    count = bigram_counts[(word1, word2)]

    previous_word_count = unigram_counts[word1]

    if previous_word_count == 0:
        return 0.0

    return count / previous_word_count


# ------------------------------------------------------------
# DISPLAY BIGRAM PROBABILITIES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BIGRAM PROBABILITIES")
print("=" * 70)

print(f"{'Bigram':35s}" f"{'Count':10s}" f"{'Probability':15s}")

print("-" * 70)

for (word1, word2), count in sorted(bigram_counts.items()):

    probability = bigram_probability(word1, word2)

    bigram_text = word1 + " " + word2

    print(f"{bigram_text:35s}" f"{count:<10d}" f"{probability:<15.6f}")


# ------------------------------------------------------------
# SENTENCE PROBABILITY
# ------------------------------------------------------------


def sentence_probability(sentence):

    words = tokenize(sentence)

    if not words:
        return 0.0

    # Add sentence boundaries
    words = ["<s>"] + words + ["</s>"]

    probability = 1.0

    print("\n" + "=" * 70)
    print("SENTENCE PROBABILITY CALCULATION")
    print("=" * 70)

    print("\nSentence:", sentence)

    print("\nBigram probabilities:")

    for i in range(len(words) - 1):

        word1 = words[i]
        word2 = words[i + 1]

        probability_value = bigram_probability(word1, word2)

        print(f"P({word2} | {word1}) = " f"{probability_value:.6f}")

        # Unsmoothed model:
        # if a bigram is unseen, probability = 0
        if probability_value == 0:

            print("\nUnseen bigram:", word1, word2)

            return 0.0

        probability *= probability_value

    return probability


# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENTER A SENTENCE")
print("=" * 70)

sentence = input("Enter a sentence to calculate its probability: ")


# ------------------------------------------------------------
# CALCULATE AND DISPLAY RESULT
# ------------------------------------------------------------

probability = sentence_probability(sentence)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("Sentence:", sentence)

print("Probability:", probability)

print("=" * 70)
