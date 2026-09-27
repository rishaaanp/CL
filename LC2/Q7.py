def minimum_edit_distance(source, target):

    m = len(source)
    n = len(target)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i * 1  # Deletion cost = 1

    for j in range(n + 1):
        dp[0][j] = j * 1  # Insertion cost = 1

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):

            # Match
            if source[i - 1] == target[j - 1]:
                cost = 0

            # Substitution
            else:
                cost = 2

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Match/Substitution
            )

    # Backtracking
    operations = []

    i = m
    j = n

    while i > 0 or j > 0:

        # Match
        if (
            i > 0
            and j > 0
            and source[i - 1] == target[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            operations.append(f"Match '{source[i - 1]}'")
            i -= 1
            j -= 1

        # Substitution
        elif (
            i > 0
            and j > 0
            and source[i - 1] != target[j - 1]
            and dp[i][j] == dp[i - 1][j - 1] + 2
        ):
            operations.append(f"Substitute '{source[i - 1]}' with '{target[j - 1]}'")
            i -= 1
            j -= 1

        # Deletion
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(f"Delete '{source[i - 1]}'")
            i -= 1

        # Insertion
        else:
            operations.append(f"Insert '{target[j - 1]}'")
            j -= 1

    operations.reverse()

    return dp[m][n], dp, operations


# Main program

source = input("Enter the first string: ")
target = input("Enter the second string: ")

distance, dp, operations = minimum_edit_distance(source, target)

print("\nMinimum Edit Distance:", distance)

print("\nEdit Operations:")

for operation in operations:
    print(operation)
