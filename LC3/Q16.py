import re
import math
from collections import Counter

# ============================================================
# Q16: TF-IDF MATRIX AND COSINE SIMILARITY
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

print("=" * 80)
print("VOCABULARY")
print("=" * 80)

print(vocabulary)

print("\nVocabulary size:", len(vocabulary))


# ------------------------------------------------------------
# CALCULATE TERM FREQUENCY (TF)
# ------------------------------------------------------------


def calculate_tf(document):

    word_counts = Counter(document)

    total_words = len(document)

    tf = {}

    for word in vocabulary:

        tf[word] = word_counts[word] / total_words

    return tf


# ------------------------------------------------------------
# CALCULATE DOCUMENT FREQUENCY (DF)
# ------------------------------------------------------------

document_frequency = {}

number_of_documents = len(tokenized_documents)

for word in vocabulary:

    count = 0

    for document in tokenized_documents:

        if word in document:
            count += 1

    document_frequency[word] = count


# ------------------------------------------------------------
# CALCULATE IDF
# ------------------------------------------------------------


def calculate_idf(word):

    df = document_frequency[word]

    return math.log(number_of_documents / df)


idf = {}

for word in vocabulary:

    idf[word] = calculate_idf(word)


# ------------------------------------------------------------
# CALCULATE TF-IDF
# ------------------------------------------------------------

tfidf_matrix = []

for document in tokenized_documents:

    tf = calculate_tf(document)

    vector = []

    for word in vocabulary:

        tfidf_value = tf[word] * idf[word]

        vector.append(tfidf_value)

    tfidf_matrix.append(vector)


# ------------------------------------------------------------
# DISPLAY IDF VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("IDF VALUES")
print("=" * 80)

for word in vocabulary:

    print(f"{word:15s} : " f"{idf[word]:.6f}")


# ------------------------------------------------------------
# DISPLAY TF-IDF MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TF-IDF MATRIX")
print("=" * 80)

print(f"{'Document':12s}", end="")

for word in vocabulary:

    print(f"{word:12s}", end="")

print()

print("-" * 80)


for i, vector in enumerate(tfidf_matrix):

    print(f"D{i + 1:<11}", end="")

    for value in vector:

        print(f"{value:<12.4f}", end="")

    print()


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_similarity(vector1, vector2):

    # Dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    # Magnitude of first vector
    magnitude1 = math.sqrt(sum(a * a for a in vector1))

    # Magnitude of second vector
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# ============================================================
# DOCUMENT SIMILARITY
# ============================================================


def document_similarity(doc1, doc2):

    if doc1 < 1 or doc1 > number_of_documents or doc2 < 1 or doc2 > number_of_documents:

        print("Invalid document number.")

        return

    vector1 = tfidf_matrix[doc1 - 1]
    vector2 = tfidf_matrix[doc2 - 1]

    similarity = cosine_similarity(vector1, vector2)

    print("\n" + "=" * 80)
    print("DOCUMENT COSINE SIMILARITY")
    print("=" * 80)

    print("Document", doc1, ":", documents[doc1 - 1])
    print("Document", doc2, ":", documents[doc2 - 1])

    print("\nCosine similarity =", round(similarity, 6))


# ============================================================
# WORD VECTORS
# ============================================================


def get_word_vector(word):

    word = word.lower()

    if word not in vocabulary:

        return None

    vector = []

    # Each word is represented by its TF-IDF
    # value across all documents.

    for document_index in range(number_of_documents):

        vector.append(tfidf_matrix[document_index][vocabulary.index(word)])

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

        print("Word not found in vocabulary:", word1)

        return

    if vector2 is None:

        print("Word not found in vocabulary:", word2)

        return

    similarity = cosine_similarity(vector1, vector2)

    print("\n" + "=" * 80)
    print("WORD COSINE SIMILARITY")
    print("=" * 80)

    print("Word 1:", word1)
    print("Vector:", vector1)

    print("\nWord 2:", word2)
    print("Vector:", vector2)

    print("\nCosine similarity =", round(similarity, 6))


# ============================================================
# INTERACTIVE MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("TF-IDF AND COSINE SIMILARITY")
    print("=" * 80)

    print("1. Show TF-IDF matrix")
    print("2. Calculate cosine similarity between two documents")
    print("3. Calculate cosine similarity between two words")
    print("4. Show training documents")
    print("5. Exit")

    choice = input("\nEnter your choice: ")

    # --------------------------------------------------------
    # OPTION 1
    # --------------------------------------------------------

    if choice == "1":

        print("\nTF-IDF Matrix:")

        print(f"{'Document':12s}", end="")

        for word in vocabulary:

            print(f"{word:12s}", end="")

        print()

        for i, vector in enumerate(tfidf_matrix):

            print(f"D{i + 1:<11}", end="")

            for value in vector:

                print(f"{value:<12.4f}", end="")

            print()

    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    elif choice == "2":

        try:

            doc1 = int(input("Enter first document number: "))

            doc2 = int(input("Enter second document number: "))

            document_similarity(doc1, doc2)

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
