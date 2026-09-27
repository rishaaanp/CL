# Finite State Automaton for English plural nouns ending in 'y'


def is_plural_y_word(word):

    word = word.lower()

    # State 0: Start state
    state = 0

    # Word must contain at least 2 characters
    if len(word) < 2:
        return False

    # Check whether word ends in "s"
    if word.endswith("s"):

        # Remove the final 's'
        singular = word[:-1]

        # The singular form must end in 'y'
        if not singular.endswith("y"):
            return False

        # Character before y
        if len(singular) < 2:
            return False

        previous = singular[-2]

        # Vowel + y -> plural is formed with s
        if previous in "aeiou":
            state = 1

        else:
            # Consonant + y cannot form plural using only s
            state = 2

    # Check for "ies"
    elif word.endswith("ies"):

        # Remove "ies" and restore "y"
        singular = word[:-3] + "y"

        # Need at least one character before y
        if len(singular) < 2:
            return False

        previous = singular[-2]

        # Consonant + y -> ies
        if previous not in "aeiou":
            state = 3

        else:
            # Vowel + y -> ies is incorrect
            state = 4

    else:
        return False

    # Accepting states
    if state in [1, 3]:
        return True

    return False


# Main program
print("FSA for English plural nouns ending in 'y'")
print("Enter words one by one.")
print("Enter 'QUIT' to stop.\n")

while True:

    word = input("Enter word: ")

    if word.upper() == "QUIT":
        break

    if is_plural_y_word(word):
        print("Accepted")
    else:
        print("Rejected")
