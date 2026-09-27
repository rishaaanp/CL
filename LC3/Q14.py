# ============================================================
# Q14: VITERBI ALGORITHM FOR POS TAGGING
# ============================================================

# States / POS tags
tags = ["NN", "VB", "JJ", "RB"]


# ------------------------------------------------------------
# TRANSITION PROBABILITIES
# P(next_tag | current_tag)
# ------------------------------------------------------------

transition = {
    "START": {"NN": 0.5, "VB": 0.25, "JJ": 0.25, "RB": 0},
    "NN": {"NN": 0.25, "VB": 0.5, "JJ": 0, "RB": 0},
    "VB": {"NN": 0.25, "VB": 0, "JJ": 0.25, "RB": 0.25},
    "JJ": {"NN": 0.75, "VB": 0, "JJ": 0.25, "RB": 0},
    "RB": {"NN": 0.25, "VB": 0, "JJ": 0.25, "RB": 0},
}


# Probability of transitioning from a tag to STOP
stop_probability = {"NN": 0.25, "VB": 0.25, "JJ": 0, "RB": 0.5}


# ------------------------------------------------------------
# EMISSION PROBABILITIES
# P(word | tag)
# ------------------------------------------------------------

emission = {
    "NN": {"time": 0.1, "flies": 0.01, "fast": 0.01},
    "VB": {"time": 0.01, "flies": 0.1, "fast": 0.01},
    "JJ": {"time": 0, "flies": 0, "fast": 0.1},
    "RB": {"time": 0, "flies": 0.1, "fast": 0.1},
}


# ------------------------------------------------------------
# INPUT SENTENCE
# ------------------------------------------------------------

sentence = ["time", "flies", "fast"]


# ------------------------------------------------------------
# VITERBI TABLES
# ------------------------------------------------------------

viterbi = []
backpointer = []


# ------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------

first_word = sentence[0]

viterbi_first = {}
backpointer_first = {}

for tag in tags:

    transition_probability = transition["START"][tag]

    emission_probability = emission[tag][first_word]

    viterbi_first[tag] = transition_probability * emission_probability

    backpointer_first[tag] = "START"


viterbi.append(viterbi_first)
backpointer.append(backpointer_first)


# ------------------------------------------------------------
# RECURSION
# ------------------------------------------------------------

for i in range(1, len(sentence)):

    word = sentence[i]

    current_viterbi = {}
    current_backpointer = {}

    for current_tag in tags:

        best_probability = 0
        best_previous_tag = None

        for previous_tag in tags:

            probability = (
                viterbi[i - 1][previous_tag]
                * transition[previous_tag][current_tag]
                * emission[current_tag][word]
            )

            if probability > best_probability:

                best_probability = probability
                best_previous_tag = previous_tag

        current_viterbi[current_tag] = best_probability
        current_backpointer[current_tag] = best_previous_tag

    viterbi.append(current_viterbi)
    backpointer.append(current_backpointer)


# ------------------------------------------------------------
# DISPLAY VITERBI TABLE
# ------------------------------------------------------------

print("=" * 70)
print("VITERBI TABLE")
print("=" * 70)

print(f"{'Word':15s}" f"{'NN':15s}" f"{'VB':15s}" f"{'JJ':15s}" f"{'RB':15s}")

print("-" * 70)

for i, word in enumerate(sentence):

    print(
        f"{word:15s}"
        f"{viterbi[i]['NN']:<15.8f}"
        f"{viterbi[i]['VB']:<15.8f}"
        f"{viterbi[i]['JJ']:<15.8f}"
        f"{viterbi[i]['RB']:<15.8f}"
    )


# ------------------------------------------------------------
# TERMINATION
# ------------------------------------------------------------

best_final_tag = None
best_final_probability = 0

for tag in tags:

    probability = viterbi[-1][tag] * stop_probability[tag]

    if probability > best_final_probability:

        best_final_probability = probability
        best_final_tag = tag


# ------------------------------------------------------------
# BACKTRACKING
# ------------------------------------------------------------

best_path = [best_final_tag]

for i in range(len(sentence) - 1, 0, -1):

    previous_tag = backpointer[i][best_path[-1]]

    best_path.append(previous_tag)

best_path.reverse()


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("\nSentence:")

for word, tag in zip(sentence, best_path):

    print(f"{word:10s} -> {tag}")


print("\nMost probable POS sequence:")
print(" -> ".join(best_path))

print("\nFinal probability:")
print(best_final_probability)

print("\nTagged sentence:")

for word, tag in zip(sentence, best_path):

    print(f"{word}/{tag}", end=" ")

print()
