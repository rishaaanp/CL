from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# ============================================================
# Q11: NEURAL SPELLING ERROR DETECTION AND CORRECTION
# ============================================================

MODEL_NAME = "vennify/t5-base-grammar-correction"

print("Loading neural model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# FUNCTION TO CORRECT A SENTENCE
# ============================================================


def correct_sentence(sentence):

    # T5 grammar-correction models generally use this prefix
    input_text = "grammar: " + sentence

    # Convert sentence into model input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

    # Generate corrected sentence
    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)

    # Convert model output back to text
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return corrected.strip()


# ============================================================
# FUNCTION TO NORMALIZE OUTPUT FOR FAIR EVALUATION
# ============================================================


def normalize(text):

    # Convert to lowercase
    text = text.lower().strip()

    # Remove punctuation at the end of the sentence
    text = re.sub(r"[.!?]+$", "", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# TEST DATA
# ============================================================

test_data = [
    ("I like to wach good movies", "I like to watch good movies"),
    ("She is a gud student", "She is a good student"),
    ("I recieved your message", "I received your message"),
    ("He is going to teh school", "He is going to the school"),
    ("This is a beautifull movie", "This is a beautiful movie"),
    ("I definately like this movie", "I definitely like this movie"),
    ("They are playing criket", "They are playing cricket"),
]


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("NEURAL MODEL EVALUATION")
print("=" * 60)

correct = 0
total = len(test_data)

for input_sentence, expected in test_data:

    # Correct the sentence using neural model
    predicted = correct_sentence(input_sentence)

    # Normalize both sentences before comparison
    normalized_predicted = normalize(predicted)
    normalized_expected = normalize(expected)

    # Compare
    if normalized_predicted == normalized_expected:
        result = "CORRECT"
        correct += 1
    else:
        result = "INCORRECT"

    # Display result
    print("\nInput    :", input_sentence)
    print("Expected :", expected)
    print("Predicted:", predicted)
    print("Result   :", result)


# ============================================================
# FINAL PERFORMANCE
# ============================================================

accuracy = (correct / total) * 100

print("\n" + "=" * 60)
print("FINAL PERFORMANCE")
print("=" * 60)

print("Correct predictions:", correct, "/", total)
print("Accuracy:", round(accuracy, 2), "%")

print("=" * 60)
