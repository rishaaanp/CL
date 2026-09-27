import re
import math
from collections import Counter

# ============================================================
# Q17: PPMI MATRIX AND COSINE SIMILARITY
# ============================================================


# ------------------------------------------------------------
# TRAINING DOCUMENTS
# ------------------------------------------------------------

documents = [
    "I like watching movies",
    "I like good movies",
    "I enjoy watching good movies",
    "She likes watching movies",
    "She likes good movies",
]


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------


def tokenize(text):

    return re.findall(r"[a-z]+", text.lower())


# Tokenize all documents
tokenized_documents = [tokenize(document) for document in documents]


# ------------------------------------------------------------
# BUILD VOCABULARY
# ------------------------------------------------------------

vocabulary = sorted(set(word for document in tokenized_documents for word in document))

number_of_documents = len(documents)

vocab_size = len(vocabulary)


# ------------------------------------------------------------
# DISPLAY VOCABULARY
# ------------------------------------------------------------

print("=" * 80)
print("VOCABULARY")
print("=" * 80)

print(vocabulary)

print("\nNumber of documents:", number_of_documents)
print("Vocabulary size:", vocab_size)


# ============================================================
# BUILD DOCUMENT-WORD CO-OCCURRENCE COUNTS
# ============================================================

# Each cell contains the number of times a word occurs
# in a particular document.

document_word_counts = {}

for document_index, words in enumerate(tokenized_documents):

    document_word_counts[document_index] = Counter(words)


# ------------------------------------------------------------
# TOTAL WORD COUNT
# ------------------------------------------------------------

total_count = sum(sum(counter.values()) for counter in document_word_counts.values())


# ------------------------------------------------------------
# DOCUMENT COUNTS
# ------------------------------------------------------------

document_counts = {}

for document_index, words in enumerate(tokenized_documents):

    document_counts[document_index] = len(words)


# ------------------------------------------------------------
# WORD COUNTS
# ------------------------------------------------------------

word_counts = Counter()

for words in tokenized_documents:

    word_counts.update(words)


# ============================================================
# CALCULATE PPMI
# ============================================================


def calculate_ppmi(document_index, word):

    # Joint count:
    # Number of times the word occurs in this document
    joint_count = document_word_counts[document_index][word]

    if joint_count == 0:
        return 0.0

    # P(document, word)
    p_document_word = joint_count / total_count

    # P(document)
    p_document = document_counts[document_index] / total_count

    # P(word)
    p_word = word_counts[word] / total_count

    # PMI
    pmi = math.log2(p_document_word / (p_document * p_word))

    # PPMI = max(PMI, 0)
    ppmi = max(pmi, 0)

    return ppmi


# ============================================================
# CREATE PPMI MATRIX
# ============================================================

ppmi_matrix = []

for document_index in range(number_of_documents):

    row = []

    for word in vocabulary:

        value = calculate_ppmi(document_index, word)

        row.append(value)

    ppmi_matrix.append(row)


# ============================================================
# DISPLAY PPMI MATRIX
# ============================================================

print("\n" + "=" * 80)
print("PPMI MATRIX")
print("=" * 80)

print(f"{'Document':12s}", end="")

for word in vocabulary:

    print(f"{word:12s}", end="")

print()

print("-" * 80)


for i, row in enumerate(ppmi_matrix):

    print(f"D{i + 1:<11}", end="")

    for value in row:

        print(f"{value:<12.4f}", end="")

    print()


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_similarity(vector1, vector2):

    # Dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    # Magnitude of vector 1
    magnitude1 = math.sqrt(sum(a * a for a in vector1))

    # Magnitude of vector 2
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:

        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# ============================================================
# DOCUMENT SIMILARITY
# ============================================================


def document_similarity(document1, document2):

    if (
        document1 < 1
        or document1 > number_of_documents
        or document2 < 1
        or document2 > number_of_documents
    ):

        print("Invalid document number.")

        return

    vector1 = ppmi_matrix[document1 - 1]

    vector2 = ppmi_matrix[document2 - 1]

    similarity = cosine_similarity(vector1, vector2)

    print("\n" + "=" * 80)
    print("DOCUMENT COSINE SIMILARITY")
    print("=" * 80)

    print("Document", document1, ":", documents[document1 - 1])

    print("Document", document2, ":", documents[document2 - 1])

    print("\nCosine similarity:", round(similarity, 6))


# ============================================================
# GET WORD VECTOR
# ============================================================


def get_word_vector(word):

    word = word.lower()

    if word not in vocabulary:

        return None

    word_index = vocabulary.index(word)

    # Column of the PPMI matrix
    vector = []

    for document_index in range(number_of_documents):

        vector.append(ppmi_matrix[document_index][word_index])

    return vector


# ============================================================
# WORD SIMILARITY
# ============================================================


def word_similarity(word1, word2):

    word1 = word1.lower()
    word2 = word2.lower()

    vector1 = get_word_vector(word1)

    vector2 = get_word_vector(word2)

    if vector1 is None:

        print("Word not found:", word1)

        return

    if vector2 is None:

        print("Word not found:", word2)

        return

    similarity = cosine_similarity(vector1, vector2)

    print("\n" + "=" * 80)
    print("WORD COSINE SIMILARITY")
    print("=" * 80)

    print("Word 1:", word1)

    print("Vector 1:", vector1)

    print("\nWord 2:", word2)

    print("Vector 2:", vector2)

    print("\nCosine similarity:", round(similarity, 6))


# ============================================================
# INTERACTIVE MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("PPMI AND COSINE SIMILARITY")
    print("=" * 80)

    print("1. Display PPMI matrix")
    print("2. Calculate cosine similarity between two documents")
    print("3. Calculate cosine similarity between two words")
    print("4. Display training documents")
    print("5. Exit")

    choice = input("\nEnter your choice: ")

    # --------------------------------------------------------
    # OPTION 1
    # --------------------------------------------------------

    if choice == "1":

        print("\nPPMI Matrix:")

        print(f"{'Document':12s}", end="")

        for word in vocabulary:

            print(f"{word:12s}", end="")

        print()

        for i, row in enumerate(ppmi_matrix):

            print(f"D{i + 1:<11}", end="")

            for value in row:

                print(f"{value:<12.4f}", end="")

            print()

    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    elif choice == "2":

        try:

            document1 = int(input("Enter first document number: "))

            document2 = int(input("Enter second document number: "))

            document_similarity(document1, document2)

        except ValueError:

            print("Please enter valid document numbers.")

    # --------------------------------------------------------
    # OPTION 3
    # --------------------------------------------------------

    elif choice == "3":

        word1 = input("Enter first word: ")

        word2 = input("Enter second word: ")

        word_similarity(word1, word2)

    # --------------------------------------------------------
    # OPTION 4
    # --------------------------------------------------------

    elif choice == "4":

        print("\n" + "=" * 80)

        print("TRAINING DOCUMENTS")

        print("=" * 80)

        for i, document in enumerate(documents):

            print(f"D{i + 1}: {document}")

    # --------------------------------------------------------
    # OPTION 5
    # --------------------------------------------------------

    elif choice == "5":

        print("\nProgram terminated.")

        break

    else:

        print("\nInvalid choice. Please try again.")
