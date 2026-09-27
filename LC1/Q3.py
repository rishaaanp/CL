import re

# Read input from file
with open("input3.txt", "r") as file:
    text = file.read()

# Tokenization pattern
pattern = r"""
    # Abbreviations: U.S.A., U.K., U.S.
    [A-Za-z]+(?:\.[A-Za-z]+)+(?:\.)?

    |

    # Hyphenated words: ice-cream, well-known
    [A-Za-z]+(?:-[A-Za-z]+)+

    |

    # Contractions ending in n't
    [A-Za-z]+n't

    |

    # Other contractions
    [A-Za-z]+(?:'s|'m|'re|'ve|'ll|'d)

    |

    # Normal words
    [A-Za-z]+

    |

    # Numbers
    \d+(?:\.\d+)?

    |

    # Punctuation and special symbols
    [^\w\s]
"""

# Find tokens
raw_tokens = re.findall(pattern, text, re.VERBOSE)

# Split contractions into two tokens
tokens = []

for token in raw_tokens:

    if re.match(r"^[A-Za-z]+n't$", token):
        # isn't -> is + n't
        tokens.append(token[:-3])
        tokens.append("n't")

    elif re.match(r"^[A-Za-z]+'(?:s|m|re|ve|ll|d)$", token):
        # I'm -> I + 'm
        # he'll -> he + 'll
        match = re.match(r"^([A-Za-z]+)('(?:s|m|re|ve|ll|d))$", token)

        tokens.append(match.group(1))
        tokens.append(match.group(2))

    else:
        tokens.append(token)


print("Tokens:")
for token in tokens:
    print(token)
