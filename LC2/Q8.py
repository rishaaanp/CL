import re
from collections import Counter

# ============================================================
# 1. TOKENIZATION
# ============================================================


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """

    return re.findall(r"[a-z]+", text.lower())


# ============================================================
# 2. READ TRAINING CORPUS
# ============================================================

with open("corpus.txt", "r") as file:
    corpus = file.read()

tokens = tokenize(corpus)

# Create vocabulary
vocabulary = set(tokens)

print("==========================================")
print("VOCABULARY")
print("==========================================")

print("Number of words in vocabulary:", len(vocabulary))
print("Vocabulary:")

for word in sorted(vocabulary):
    print(word)


# ============================================================
# 3. CREATE BIGRAM FREQUENCY TABLE
# ============================================================

# Add sentence boundary markers
bigram_tokens = ["<s>"] + tokens + ["</s>"]

bigram_counts = Counter()

for i in range(len(bigram_tokens) - 1):

    bigram = (bigram_tokens[i], bigram_tokens[i + 1])

    bigram_counts[bigram] += 1


print("\n==========================================")
print("BIGRAM FREQUENCY TABLE")
print("==========================================")

for bigram, count in bigram_counts.items():

    print(bigram[0], "->", bigram[1], ":", count)


# ============================================================
# 4. WORD FREQUENCY TABLE
# ============================================================

word_counts = Counter(tokens)

total_words = len(tokens)

# Number of unique words
V = len(vocabulary)


# ============================================================
# 5. BIGRAM PROBABILITY
# ============================================================


def bigram_probability(w1, w2):
    """
    Calculate P(w2 | w1) using add-one smoothing.

    P(w2 | w1) =
    (count(w1,w2) + 1) /
    (count(w1) + V)
    """

    bigram_count = bigram_counts.get((w1, w2), 0)

    word_count = word_counts.get(w1, 0)

    probability = (bigram_count + 1) / (word_count + V)

    return probability


# ============================================================
# 6. EDIT DISTANCE
# ============================================================


def edit_distance(word1, word2):

    m = len(word1)
    n = len(word2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialization
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


# ============================================================
# 7. GENERATE CANDIDATES WITH EDIT DISTANCE 1
# ============================================================


def generate_candidates(word):

    candidates = set()

    for vocabulary_word in vocabulary:

        if edit_distance(word, vocabulary_word) == 1:
            candidates.add(vocabulary_word)

    return candidates


# ============================================================
# 8. CALCULATE SENTENCE PROBABILITY
# ============================================================


def sentence_probability(sentence):

    words = tokenize(sentence)

    if not words:
        return 0

    words = ["<s>"] + words + ["</s>"]

    probability = 1.0

    for i in range(len(words) - 1):

        p = bigram_probability(words[i], words[i + 1])

        probability *= p

    return probability


# ============================================================
# 9. FIND NON-WORD ERRORS
# ============================================================

with open("input8.txt", "r") as file:
    input_text = file.read()

input_words = tokenize(input_text)

errors = []

for word in input_words:

    if word not in vocabulary:
        errors.append(word)


print("\n==========================================")
print("NON-WORD SPELLING ERRORS")
print("==========================================")

if len(errors) == 0:

    print("No spelling errors found.")

else:

    for error in errors:
        print(error)


# ============================================================
# 10. SPELL CORRECTION
# ============================================================

print("\n==========================================")
print("SPELL CORRECTION")
print("==========================================")

corrected_sentence = input_text

for error in errors:

    # Generate candidate words
    candidates = generate_candidates(error)

    print("\nMisspelled word:", error)

    print("Candidates:")

    if not candidates:
        print("No candidates found.")
        continue

    for candidate in sorted(candidates):
        print(candidate)

    # Store probabilities
    candidate_probabilities = {}

    # Replace error with every candidate
    for candidate in candidates:

        candidate_sentence = re.sub(
            r"\b" + re.escape(error) + r"\b", candidate, corrected_sentence, count=1
        )

        probability = sentence_probability(candidate_sentence)

        candidate_probabilities[candidate] = probability

    # Find candidate with highest probability
    best_candidate = max(candidate_probabilities, key=candidate_probabilities.get)

    print("\nCandidate probabilities:")

    for candidate, probability in sorted(
        candidate_probabilities.items(), key=lambda x: x[1], reverse=True
    ):

        print(candidate, ":", probability)

    print("Best candidate:", best_candidate)

    # Replace error with best candidate
    corrected_sentence = re.sub(
        r"\b" + re.escape(error) + r"\b", best_candidate, corrected_sentence, count=1
    )


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print("Original sentence:")
print(input_text)

print("\nCorrected sentence:")
print(corrected_sentence)
