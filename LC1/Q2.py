import re


def eliza_response(user_input):

    user_input = user_input.lower().strip()

    # Substitution rules
    substitutions = [
        (r"i like (.*)", "What do you like about {0}?"),
        (r"i love (.*)", "What do you love about {0}?"),
        (r"i hate (.*)", "What don't you like about {0}?"),
        (r"i watched (.*)", "What did you think about {0}?"),
        (r"i saw (.*)", "Did you enjoy {0}?"),
        (r"my favorite movie is (.*)", "What makes {0} your favorite movie?"),
        (r"my favorite actor is (.*)", "What do you like about {0}?"),
        (r"my favorite actress is (.*)", "What do you like about {0}?"),
        (r"i am watching (.*)", "How are you finding {0} so far?"),
        (r"i want to watch (.*)", "Why do you want to watch {0}?"),
        (r"i want to see (.*)", "Why do you want to see {0}?"),
        (r"i prefer (.*)", "Why do you prefer {0}?"),
        (r"i feel (.*)", "What movie makes you feel {0}?"),
        (r"do you like (.*)", "What do you like about {0}?"),
        (r"have you watched (.*)", "What did you like about {0}?"),
        (r"recommend (.*)", "What kind of movies similar to {0} do you enjoy?"),
        (r"because (.*)", "Is that why you enjoy movies?"),
    ]

    # Match input with substitution rules
    for pattern, response in substitutions:

        match = re.match(pattern, user_input)

        if match:
            return response.format(*match.groups())

    # Default responses
    default_responses = [
        "Tell me more about that movie.",
        "What kind of movies do you usually enjoy?",
        "Who is your favorite actor?",
        "What is your favorite movie?",
        "Do you prefer watching movies at home or in a theatre?",
        "Which movie genre do you like the most?",
    ]

    # Select a response based on input length
    return default_responses[len(user_input) % len(default_responses)]


# Main program
print("ELIZA: Hello! I am a movie chatbot.")
print("ELIZA: Let's talk about movies!")
print("ELIZA: Type 'BYE BYE' to end the conversation.\n")

while True:

    user_input = input("You: ")

    # Exit condition
    if user_input.strip().upper() == "BYE BYE":
        print("ELIZA: Goodbye! Enjoy your next movie!")
        break

    response = eliza_response(user_input)

    print("ELIZA:", response)
