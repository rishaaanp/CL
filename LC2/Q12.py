import re
import random
import math
from collections import Counter

# ============================================================
# Q12: UNIGRAM AND BIGRAM LANGUAGE MODELS
# ============================================================


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------


def tokenize(text):
    """
    Convert text to lowercase and extract words.
    """
    return re.findall(r"[a-z]+", text.lower())


# ------------------------------------------------------------
# READ TRAINING CORPUS
# ------------------------------------------------------------

with open("train.txt", "r", encoding="utf-8") as file:
    train_text = file.read()

train_sentences = [line.strip() for line in train_text.splitlines() if line.strip()]


# ------------------------------------------------------------
# BUILD TRAINING TOKENS
# ------------------------------------------------------------

train_tokens = tokenize(train_text)

# Unigram counts
unigram_counts = Counter(train_tokens)

# Total number of words
total_words = len(train_tokens)

# Vocabulary
vocabulary = set(train_tokens)

vocab_size = len(vocabulary)


# ------------------------------------------------------------
# BUILD BIGRAM COUNTS
# ------------------------------------------------------------

bigram_counts = Counter()
bigram_context_counts = Counter()

for sentence in train_sentences:

    words = tokenize(sentence)

    # Add sentence boundary symbols
    words = ["<s>"] + words + ["</s>"]

    for i in range(len(words) - 1):

        w1 = words[i]
        w2 = words[i + 1]

        bigram_counts[(w1, w2)] += 1
        bigram_context_counts[w1] += 1


# ------------------------------------------------------------
# UNIGRAM PROBABILITY
# ------------------------------------------------------------


def unigram_probability(word):

    word = word.lower()

    if word not in vocabulary:
        return 0.0

    return unigram_counts[word] / total_words


# ------------------------------------------------------------
# BIGRAM PROBABILITY
# ------------------------------------------------------------


def bigram_probability(w1, w2):

    w1 = w1.lower()
    w2 = w2.lower()

    count = bigram_counts[(w1, w2)]

    context_count = bigram_context_counts[w1]

    if context_count == 0:
        return 0.0

    return count / context_count


# ============================================================
# PART (a)
# COMPUTE UNSMOOTHED UNIGRAMS AND BIGRAMS
# ============================================================


def display_counts_and_probabilities():

    print("\n" + "=" * 70)
    print("UNIGRAM MODEL")
    print("=" * 70)

    print(f"{'Word':20s} {'Count':10s} {'Probability':15s}")

    for word, count in sorted(unigram_counts.items()):

        probability = count / total_words

        print(f"{word:20s} " f"{count:<10d} " f"{probability:.6f}")

    print("\n" + "=" * 70)
    print("BIGRAM MODEL")
    print("=" * 70)

    print(f"{'Bigram':35s} " f"{'Count':10s} " f"{'Probability':15s}")

    for (w1, w2), count in sorted(bigram_counts.items()):

        probability = bigram_probability(w1, w2)

        print(f"{(w1 + ' ' + w2):35s} " f"{count:<10d} " f"{probability:.6f}")


# ============================================================
# PART (b)
# COMPUTE PROBABILITY OF USER-ENTERED TEXT
# ============================================================


def unigram_text_probability(text):

    words = tokenize(text)

    if not words:
        return 0.0

    probability = 1.0

    for word in words:

        p = unigram_probability(word)

        if p == 0:
            return 0.0

        probability *= p

    return probability


def bigram_text_probability(text):

    words = tokenize(text)

    if not words:
        return 0.0

    words = ["<s>"] + words + ["</s>"]

    probability = 1.0

    for i in range(len(words) - 1):

        p = bigram_probability(words[i], words[i + 1])

        # Unsmoothed model:
        # unseen bigram = probability 0
        if p == 0:
            return 0.0

        probability *= p

    return probability


def calculate_text_probability():

    print("\n" + "=" * 70)
    print("TEXT PROBABILITY")
    print("=" * 70)

    text = input("Enter a sentence: ")

    unigram_p = unigram_text_probability(text)
    bigram_p = bigram_text_probability(text)

    print("\nSentence:", text)

    print("\nUnigram probability:")
    print(unigram_p)

    print("\nBigram probability:")
    print(bigram_p)

    if unigram_p == 0:
        print("\nUnigram model: Probability is 0")
        print("At least one word is unseen in the training corpus.")

    if bigram_p == 0:
        print("\nBigram model: Probability is 0")
        print("At least one bigram is unseen in the training corpus.")


# ============================================================
# PART (c-i)
# GENERATE RANDOM SENTENCE USING UNIGRAM MODEL
# ============================================================


def generate_unigram_sentence(max_words=10):

    words = []

    # Create list according to frequency.
    # More frequent words get selected more often.
    population = list(unigram_counts.keys())
    weights = list(unigram_counts.values())

    for _ in range(max_words):

        word = random.choices(population, weights=weights, k=1)[0]

        words.append(word)

    return " ".join(words)


# ============================================================
# PART (c-ii)
# GENERATE RANDOM SENTENCE USING BIGRAM MODEL
# ============================================================


def generate_bigram_sentence(max_words=10):

    sentence = []

    current_word = "<s>"

    for _ in range(max_words):

        # Find words that can follow current_word
        candidates = []
        weights = []

        for (w1, w2), count in bigram_counts.items():

            if w1 == current_word and w2 != "</s>":

                candidates.append(w2)
                weights.append(count)

        # If no continuation exists
        if not candidates:
            break

        # Select next word according to bigram frequency
        next_word = random.choices(candidates, weights=weights, k=1)[0]

        sentence.append(next_word)

        current_word = next_word

        # Check whether the sentence can end
        end_count = bigram_counts.get((current_word, "</s>"), 0)

        if end_count > 0:

            # Give the model a chance to stop
            if random.random() < 0.4:
                break

    return " ".join(sentence)


def generate_sentences():

    print("\n" + "=" * 70)
    print("RANDOM SENTENCE GENERATION")
    print("=" * 70)

    print("\nUNIGRAM MODEL SENTENCES")
    print("-" * 70)

    for i in range(5):

        sentence = generate_unigram_sentence()

        print(f"{i + 1}. {sentence}")

    print("\nBIGRAM MODEL SENTENCES")
    print("-" * 70)

    for i in range(5):

        sentence = generate_bigram_sentence()

        print(f"{i + 1}. {sentence}")


# ============================================================
# PART (d)
# PERPLEXITY
# ============================================================


def unigram_perplexity(test_tokens):
    """
    Unsmoothed unigram perplexity.

    If an unseen word occurs:
        probability = 0
        perplexity = infinity
    """

    N = len(test_tokens)

    if N == 0:
        return 0.0

    log_probability = 0.0

    for word in test_tokens:

        p = unigram_probability(word)

        if p == 0:
            return float("inf")

        log_probability += math.log(p)

    perplexity = math.exp(-log_probability / N)

    return perplexity


def bigram_perplexity(test_sentences):
    """
    Unsmoothed bigram perplexity.
    """

    log_probability = 0.0
    N = 0

    for sentence in test_sentences:

        words = tokenize(sentence)

        words = ["<s>"] + words + ["</s>"]

        for i in range(len(words) - 1):

            w1 = words[i]
            w2 = words[i + 1]

            p = bigram_probability(w1, w2)

            if p == 0:
                return float("inf")

            log_probability += math.log(p)

            N += 1

    if N == 0:
        return 0.0

    perplexity = math.exp(-log_probability / N)

    return perplexity


def calculate_perplexity():

    print("\n" + "=" * 70)
    print("PERPLEXITY ON TEST SET")
    print("=" * 70)

    with open("test.txt", "r", encoding="utf-8") as file:
        test_text = file.read()

    test_sentences = [line.strip() for line in test_text.splitlines() if line.strip()]

    test_tokens = tokenize(test_text)

    unigram_pp = unigram_perplexity(test_tokens)

    bigram_pp = bigram_perplexity(test_sentences)

    print("\nTest set:")
    print(test_text)

    print("\nUnigram perplexity:")

    if math.isinf(unigram_pp):
        print("Infinity")
        print("(At least one test word was unseen in training.)")
    else:
        print(round(unigram_pp, 4))

    print("\nBigram perplexity:")

    if math.isinf(bigram_pp):
        print("Infinity")
        print("(At least one test bigram was unseen in training.)")
    else:
        print(round(bigram_pp, 4))

    print("\nInterpretation:")
    print("Lower perplexity means the language model " "predicts the test data better.")


# ============================================================
# MAIN MENU
# ============================================================


def main():

    while True:

        print("\n")
        print("=" * 70)
        print("LANGUAGE MODEL PROGRAM")
        print("=" * 70)

        print("1. Display unigram and bigram counts/probabilities")
        print("2. Compute probability of entered text")
        print("3. Generate random sentences")
        print("4. Compute test-set perplexity")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            display_counts_and_probabilities()

        elif choice == "2":

            calculate_text_probability()

        elif choice == "3":

            generate_sentences()

        elif choice == "4":

            calculate_perplexity()

        elif choice == "5":

            print("\nProgram terminated.")
            break

        else:

            print("\nInvalid choice. Please try again.")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
