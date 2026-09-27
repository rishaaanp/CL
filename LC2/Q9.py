import re
from collections import Counter

# ============================================================
# 1. TOKENIZATION
# ============================================================


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# ============================================================
# 2. READ CORPUS
# ============================================================

with open("corpus.txt", "r") as file:
    corpus = file.read()

tokens = tokenize(corpus)

# Vocabulary
vocabulary = set(tokens)

# Word frequencies
word_counts = Counter(tokens)

total_words = len(tokens)
V = len(vocabulary)


# ============================================================
# 3. BIGRAM FREQUENCY TABLE
# ============================================================

bigram_tokens = ["<s>"] + tokens + ["</s>"]

bigram_counts = Counter()

for i in range(len(bigram_tokens) - 1):

    bigram = (bigram_tokens[i], bigram_tokens[i + 1])

    bigram_counts[bigram] += 1


# ============================================================
# 4. BIGRAM PROBABILITY
# ============================================================


def bigram_probability(w1, w2):

    bigram_count = bigram_counts.get((w1, w2), 0)

    word_count = word_counts.get(w1, 0)

    # Add-one smoothing
    probability = (bigram_count + 1) / (word_count + V)

    return probability


# ============================================================
# 5. SENTENCE PROBABILITY
# ============================================================


def sentence_probability(sentence):

    words = tokenize(sentence)

    if not words:
        return 0

    words = ["<s>"] + words + ["</s>"]

    probability = 1.0

    for i in range(len(words) - 1):

        probability *= bigram_probability(words[i], words[i + 1])

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

    # Dynamic programming
    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]


# ============================================================
# 7. GENERATE EDIT-DISTANCE-1 CANDIDATES
# ============================================================


def generate_candidates(error):

    candidates = set()

    for word in vocabulary:

        if edit_distance(error, word) == 1:
            candidates.add(word)

    return candidates


# ============================================================
# 8. NOISY CHANNEL PROBABILITY
# ============================================================


def channel_probability(error, candidate):

    distance = edit_distance(candidate, error)

    # We only consider candidates at edit distance 1
    if distance != 1:
        return 0

    # Simple channel model:
    #
    # insertion/deletion/substitution
    # are assumed equally likely.
    #
    # A smaller word gets a slightly higher
    # probability because there are fewer
    # possible positions for an error.

    return 1 / (len(candidate) * 3)


# ============================================================
# 9. NOISY CHANNEL SCORE
# ============================================================


def noisy_channel_score(error, candidate):

    # P(error | candidate)
    channel = channel_probability(error, candidate)

    # P(candidate)
    prior = word_counts[candidate] / total_words

    # P(candidate | error)
    score = channel * prior

    return score


# ============================================================
# 10. READ INPUT
# ============================================================

with open("input8.txt", "r") as file:
    input_text = file.read()

input_words = tokenize(input_text)


# ============================================================
# 11. FIND NON-WORD ERRORS
# ============================================================

errors = []

for word in input_words:

    if word not in vocabulary:
        errors.append(word)


print("=" * 60)
print("NON-WORD SPELLING ERRORS")
print("=" * 60)

if not errors:

    print("No spelling errors found.")

else:

    for error in errors:
        print(error)


# ============================================================
# 12. COMPARE BIGRAM LM AND NOISY CHANNEL
# ============================================================

corrected_bigram = input_text
corrected_noisy = input_text


for error in errors:

    print("\n" + "=" * 60)
    print("MISSPELLED WORD:", error)
    print("=" * 60)

    candidates = generate_candidates(error)

    print("\nCandidates at edit distance 1:")

    if not candidates:

        print("No candidates found.")
        continue

    for candidate in sorted(candidates):
        print(candidate)

    # --------------------------------------------------------
    # BIGRAM LM
    # --------------------------------------------------------

    bigram_scores = {}

    for candidate in candidates:

        candidate_sentence = re.sub(
            r"\b" + re.escape(error) + r"\b", candidate, corrected_bigram, count=1
        )

        score = sentence_probability(candidate_sentence)

        bigram_scores[candidate] = score

    best_bigram = max(bigram_scores, key=bigram_scores.get)

    # --------------------------------------------------------
    # NOISY CHANNEL
    # --------------------------------------------------------

    noisy_scores = {}

    for candidate in candidates:

        score = noisy_channel_score(error, candidate)

        noisy_scores[candidate] = score

    best_noisy = max(noisy_scores, key=noisy_scores.get)

    # --------------------------------------------------------
    # DISPLAY BIGRAM SCORES
    # --------------------------------------------------------

    print("\nBigram LM scores:")

    for candidate, score in sorted(
        bigram_scores.items(), key=lambda x: x[1], reverse=True
    ):

        print(candidate, "->", score)

    print("\nBigram LM correction:", best_bigram)

    # --------------------------------------------------------
    # DISPLAY NOISY CHANNEL SCORES
    # --------------------------------------------------------

    print("\nNoisy Channel scores:")

    for candidate, score in sorted(
        noisy_scores.items(), key=lambda x: x[1], reverse=True
    ):

        print(candidate, "->", score)

    print("\nNoisy Channel correction:", best_noisy)

    # --------------------------------------------------------
    # UPDATE SENTENCES
    # --------------------------------------------------------

    corrected_bigram = re.sub(
        r"\b" + re.escape(error) + r"\b", best_bigram, corrected_bigram, count=1
    )

    corrected_noisy = re.sub(
        r"\b" + re.escape(error) + r"\b", best_noisy, corrected_noisy, count=1
    )


# ============================================================
# 13. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nOriginal:")
print(input_text)

print("\nBigram LM:")
print(corrected_bigram)

print("\nNoisy Channel:")
print(corrected_noisy)
