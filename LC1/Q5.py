def plural_fst(word):

    # Remove word boundary
    word = word.replace("#", "")

    # Split lexical form at morpheme boundary
    stem, suffix = word.split("^")

    output = ""

    # FST transitions for the stem
    for ch in stem:
        output += ch

    # FST transition at morpheme boundary
    if stem[-1] in "xsz":
        output += "e"

    # Read plural morpheme
    output += suffix

    return output


# Main program
print("Finite State Transducer for English Plural Formation")
print("Enter lexical forms such as fox^s# or boy^s#")
print("Enter QUIT to stop.\n")

while True:

    word = input("Input: ")

    if word.upper() == "QUIT":
        break

    print("Output:", plural_fst(word))
