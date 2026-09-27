from collections import Counter

# ---------------------------------------------------------
# Step 1: Create initial vocabulary
# ---------------------------------------------------------

corpus = """
low lower lowest
new newer newest
wide wider widest
low lower new
"""

# Convert corpus into words
words = corpus.lower().split()

# Count frequency of each word
word_frequency = Counter(words)

print("Initial Word Frequencies:")
for word, freq in word_frequency.items():
    print(word, ":", freq)


# ---------------------------------------------------------
# Step 2: Represent each word as characters
# ---------------------------------------------------------

vocab = {}

for word, freq in word_frequency.items():
    # Add </w> to represent the end of a word
    symbols = tuple(list(word) + ["</w>"])
    vocab[symbols] = freq

print("\nInitial Vocabulary:")
for symbols, freq in vocab.items():
    print(" ".join(symbols), ":", freq)


# ---------------------------------------------------------
# Function to count adjacent symbol pairs
# ---------------------------------------------------------


def get_pair_frequencies(vocab):

    pairs = Counter()

    for symbols, freq in vocab.items():

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq

    return pairs


# ---------------------------------------------------------
# Function to merge the most frequent pair
# ---------------------------------------------------------


def merge_pair(vocab, best_pair):

    new_vocab = {}

    for symbols, freq in vocab.items():

        new_symbols = []
        i = 0

        while i < len(symbols):

            # If current and next symbol form best pair
            if (
                i < len(symbols) - 1
                and symbols[i] == best_pair[0]
                and symbols[i + 1] == best_pair[1]
            ):
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2

            else:
                new_symbols.append(symbols[i])
                i += 1

        new_vocab[tuple(new_symbols)] = freq

    return new_vocab


# ---------------------------------------------------------
# Step 3: Perform BPE merges
# ---------------------------------------------------------

num_merges = 10

merges = []

print("\n" + "=" * 60)
print("BPE VOCABULARY CREATION")
print("=" * 60)

for iteration in range(1, num_merges + 1):

    pair_frequencies = get_pair_frequencies(vocab)

    if not pair_frequencies:
        break

    # Find the most frequent pair
    best_pair, best_frequency = pair_frequencies.most_common(1)[0]

    print("\nStep", iteration)
    print("-" * 40)

    print("Pair frequencies:")

    for pair, freq in pair_frequencies.most_common():
        print("(" + pair[0] + ", " + pair[1] + ") :", freq)

    print("\nMost frequent pair:", best_pair, "Frequency:", best_frequency)

    # Merge the pair
    vocab = merge_pair(vocab, best_pair)

    merges.append(best_pair)

    print("\nVocabulary after merge:")

    for symbols, freq in vocab.items():
        print(" ".join(symbols), ":", freq)


# ---------------------------------------------------------
# Step 4: Display learned BPE merges
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LEARNED BPE MERGES")
print("=" * 60)

for i, pair in enumerate(merges, 1):
    print(i, ":", pair[0], "+", pair[1], "->", pair[0] + pair[1])


# ---------------------------------------------------------
# Step 5: BPE Tokenization
# ---------------------------------------------------------


def tokenize_word(word, merges):

    # Start with individual characters
    symbols = list(word) + ["</w>"]

    # Apply learned merges in the same order
    for merge in merges:

        new_symbols = []
        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == merge[0]
                and symbols[i + 1] == merge[1]
            ):
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2

            else:
                new_symbols.append(symbols[i])
                i += 1

        symbols = new_symbols

    return symbols


# ---------------------------------------------------------
# Step 6: Test tokenizer
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BPE TOKENIZATION")
print("=" * 60)

test_words = ["low", "lower", "lowest", "new", "newest", "wider", "widest"]

for word in test_words:

    tokens = tokenize_word(word, merges)

    # Remove end-of-word marker from display
    tokens = [token.replace("</w>", "") for token in tokens]

    print(word, "->", tokens)
