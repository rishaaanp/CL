import re

# Read input from file
with open("input.txt", "r") as file:
    text = file.read()

# a. Two consecutive repeated words
pattern_a = r"\b([A-Za-z]+)\s+\1\b"

print("a. Consecutive repeated words:")
matches_a = re.findall(pattern_a, text)

for word in matches_a:
    print(word, word)


# b. Lines starting with an integer and ending with a word
pattern_b = r"^\s*\d+.*\b[A-Za-z]+\s*$"

print("\nb. Lines starting with an integer and ending with a word:")

for line in text.splitlines():
    if re.search(pattern_b, line):
        print(line)


# c. Strings containing both the words "grotto" and "raven"
pattern_c = r"\b(?=.*\bgrotto\b)(?=.*\braven\b).*"

print("\nc. Lines containing both 'grotto' and 'raven':")

for line in text.splitlines():
    if re.search(pattern_c, line, re.IGNORECASE):
        print(line)


# d. First word of an English sentence
pattern_d = r'(?:^|[.!?]\s+)[\'"(\[]*\b([A-Za-z]+)\b'

print("\nd. First word of each sentence:")

matches_d = re.findall(pattern_d, text)

for word in matches_d:
    print(word)
