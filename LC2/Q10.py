import re
import math
from collections import Counter

from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. CONFUSION SETS
# ============================================================

CONFUSION_SETS = [
    {"write", "right", "rite"},
    {"peace", "piece"},
    {"their", "there", "they're"},
]

# Create lookup dictionary
WORD_TO_SET = {}

for confusion_set in CONFUSION_SETS:
    for word in confusion_set:
        WORD_TO_SET[word] = confusion_set


# ============================================================
# 2. TRAINING DATA
# ============================================================

training_sentences = [
    # write / right / rite
    "I want to write a letter",
    "Please write your name",
    "I will write a report",
    "She likes to write stories",
    "He learned to write well",
    "You are right about this",
    "That is the right answer",
    "Take the right turn",
    "You chose the right option",
    "This is the right place",
    "The priest performed the rite",
    "The wedding rite was beautiful",
    "They followed an ancient rite",
    # peace / piece
    "We all want peace",
    "The country needs peace",
    "They worked for peace",
    "The treaty brought peace",
    "Everyone deserves peace",
    "I ate a piece of cake",
    "Give me a piece of paper",
    "She broke a piece of glass",
    "He bought a piece of furniture",
    "I need one piece of information",
    # their / there / they're
    "They forgot their books",
    "The students brought their bags",
    "I like their new house",
    "They finished their work",
    "The children cleaned their room",
    "The book is there",
    "Put the bag there",
    "She lives there",
    "Your phone is over there",
    "There is a problem",
    "They're going to school",
    "They're very happy",
    "They're watching a movie",
    "I think they're ready",
    "They're coming tomorrow",
]


# ============================================================
# 3. TOKENIZATION
# ============================================================


def tokenize(sentence):

    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())


# ============================================================
# 4. CREATE TRAINING CORPUS STATISTICS
# ============================================================

all_tokens = []

for sentence in training_sentences:
    all_tokens.extend(tokenize(sentence))


unigram_counts = Counter(all_tokens)

bigram_counts = Counter()

for sentence in training_sentences:

    tokens = ["<s>"] + tokenize(sentence) + ["</s>"]

    for i in range(len(tokens) - 1):

        bigram_counts[(tokens[i], tokens[i + 1])] += 1


total_words = len(all_tokens)
vocabulary_size = len(unigram_counts)


# ============================================================
# 5. BIGRAM LOG PROBABILITY
# ============================================================


def bigram_log_probability(word1, word2):

    count_bigram = bigram_counts.get((word1, word2), 0)

    count_word1 = unigram_counts.get(word1, 0)

    # Add-one smoothing
    probability = (count_bigram + 1) / (count_word1 + vocabulary_size)

    return math.log(probability)


# ============================================================
# 6. CREATE FEATURES
# ============================================================


def extract_features(candidate, previous_word, next_word):

    # --------------------------------------------------------
    # Feature 1: Previous-word bigram probability
    # --------------------------------------------------------

    previous_bigram = bigram_log_probability(previous_word, candidate)

    # --------------------------------------------------------
    # Feature 2: Next-word bigram probability
    # --------------------------------------------------------

    next_bigram = bigram_log_probability(candidate, next_word)

    # --------------------------------------------------------
    # Feature 3: Unigram log probability
    # --------------------------------------------------------

    unigram_probability = (unigram_counts.get(candidate, 0) + 1) / (
        total_words + vocabulary_size
    )

    unigram_log_probability = math.log(unigram_probability)

    # --------------------------------------------------------
    # Feature 4: Candidate identity
    # --------------------------------------------------------

    # These features allow the model to learn
    # candidate-specific behavior.

    is_write = int(candidate == "write")
    is_right = int(candidate == "right")
    is_rite = int(candidate == "rite")

    is_peace = int(candidate == "peace")
    is_piece = int(candidate == "piece")

    is_their = int(candidate == "their")
    is_there = int(candidate == "there")
    is_theyre = int(candidate == "they're")

    return [
        previous_bigram,
        next_bigram,
        unigram_log_probability,
        is_write,
        is_right,
        is_rite,
        is_peace,
        is_piece,
        is_their,
        is_there,
        is_theyre,
    ]


# ============================================================
# 7. CREATE TRAINING EXAMPLES
# ============================================================

X = []
y = []


for sentence in training_sentences:

    tokens = tokenize(sentence)

    for i, correct_word in enumerate(tokens):

        # Check whether this word belongs
        # to one of our confusion sets.

        if correct_word not in WORD_TO_SET:
            continue

        confusion_set = WORD_TO_SET[correct_word]

        # Previous word
        if i > 0:
            previous_word = tokens[i - 1]
        else:
            previous_word = "<s>"

        # Next word
        if i < len(tokens) - 1:
            next_word = tokens[i + 1]
        else:
            next_word = "</s>"

        # Generate an example for every
        # candidate in the confusion set.

        for candidate in confusion_set:

            features = extract_features(candidate, previous_word, next_word)

            X.append(features)

            # Correct candidate = 1
            # Incorrect candidate = 0

            if candidate == correct_word:
                y.append(1)
            else:
                y.append(0)


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

print("=" * 60)
print("LOGISTIC REGRESSION MODEL TRAINED")
print("=" * 60)

print("Training examples:", len(X))
print("Number of features:", len(X[0]))


# ============================================================
# 9. FIND CANDIDATES
# ============================================================


def get_candidates(word):

    return WORD_TO_SET.get(word, set())


# ============================================================
# 10. PREDICT BEST CANDIDATE
# ============================================================


def predict_candidate(candidate, previous_word, next_word):

    features = extract_features(candidate, previous_word, next_word)

    probability = model.predict_proba([features])[0][1]

    return probability


# ============================================================
# 11. PROCESS INPUT
# ============================================================

input_text = input("\nEnter a sentence:\n")

tokens = tokenize(input_text)

corrected_tokens = tokens.copy()


# ============================================================
# 12. SCAN INPUT TOKENS
# ============================================================

for i, word in enumerate(tokens):

    # Is this word in a confusion set?
    if word not in WORD_TO_SET:
        continue

    candidates = get_candidates(word)

    # Previous context
    if i > 0:
        previous_word = tokens[i - 1]
    else:
        previous_word = "<s>"

    # Next context
    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
    else:
        next_word = "</s>"

    print("\n" + "=" * 60)
    print("TARGET WORD:", word)
    print("=" * 60)

    print("Previous word:", previous_word)

    print("Next word:", next_word)

    print("\nCandidate probabilities:")

    candidate_scores = {}

    # --------------------------------------------------------
    # Evaluate every candidate
    # --------------------------------------------------------

    for candidate in candidates:

        probability = predict_candidate(candidate, previous_word, next_word)

        candidate_scores[candidate] = probability

        print(candidate, "->", round(probability, 6))

    # --------------------------------------------------------
    # Select highest-scoring candidate
    # --------------------------------------------------------

    best_candidate = max(candidate_scores, key=candidate_scores.get)

    print("\nSelected candidate:", best_candidate)

    # Replace word
    corrected_tokens[i] = best_candidate


# ============================================================
# 13. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Original:")
print(input_text)

print("\nCorrected:")
print(" ".join(corrected_tokens))
