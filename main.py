import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random


def download_nltk_data():
    nltk.download("brown")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")


def load_data():
    return brown.sents()


def preprocess_data(data: list):
    # concatenate all sentences into a single string
    text = " ".join([" ".join(sentence) for sentence in data])

    # tokenize the text into words
    words = word_tokenize(text.lower())

    # remove stop words and non-alphabetic tokens
    # stop_words = set(stopwords.words("english"))
    # words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [word for word in words if word.isalpha()]

    return words


def build_vocab(data: list):
    # data is now a list of words, so we can directly create a set from it
    return set(data)


def build_ngram_model(data: list, n: int = 4):
    """
    Build an n-gram model from the preprocessed data with smoothing
    Returns a dictionary of (n-1)-gram to next word probabilities
    """
    ngram_counts = defaultdict(lambda: defaultdict(int))
    context_counts = defaultdict(int)
    word_counts = defaultdict(int)

    # Count n-grams and individual words
    for i in range(len(data) - n + 1):
        context = tuple(data[i : i + n - 1])
        next_word = data[i + n - 1]
        ngram_counts[context][next_word] += 1
        context_counts[context] += 1
        word_counts[next_word] += 1

    # Add-one smoothing
    vocab_size = len(word_counts)

    # Convert counts to probabilities with smoothing
    ngram_probs = defaultdict(lambda: defaultdict(float))
    for context in ngram_counts:
        for word in ngram_counts[context]:
            # Add-one smoothing formula
            ngram_probs[context][word] = (ngram_counts[context][word] + 1) / (
                context_counts[context] + vocab_size
            )

    return ngram_probs, word_counts


def predict_next_word(model, context: tuple, top_k: int = 5):
    """
    Predict the next word given a context
    Returns top k most likely next words with their probabilities
    """
    if context not in model:
        return []

    # Get probabilities for all possible next words
    word_probs = model[context]

    # Sort by probability and get top k
    sorted_probs = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:top_k]


def generate_text(model, seed_text: str, num_words: int = 10):
    """
    Generate text starting from a seed text
    """
    words = word_tokenize(seed_text.lower())
    generated = words.copy()

    for _ in range(num_words):
        # Get the last n-1 words as context
        context = tuple(generated[-(len(words) - 1) :])
        next_word_probs = predict_next_word(model, context)

        if not next_word_probs:
            # If no predictions found, try with shorter context
            if len(context) > 1:
                context = context[1:]  # Remove oldest word
                next_word_probs = predict_next_word(model, context)
            else:
                print(
                    f"Warning: Could not find predictions for context '{context}'. Stopping generation."
                )
                break

        if not next_word_probs:  # Double check after backoff
            print(
                "Warning: Could not find predictions even with shorter context. Stopping generation."
            )
            break

        # Choose next word based on probabilities
        words, probs = zip(*next_word_probs)
        next_word = random.choices(words, weights=probs, k=1)[0]
        generated.append(next_word)

    return " ".join(generated)


def main():
    print("Downloading nltk data...")
    download_nltk_data()
    print("Loading data...")
    data = load_data()
    print("Preprocessing data...")
    data = preprocess_data(data)
    print("Building vocabulary...")
    vocab = build_vocab(data)
    print("Vocabulary size:", len(vocab))

    print("\nBuilding n-gram model...")
    ngram_model, word_counts = build_ngram_model(data, n=4)  # Using 4-grams

    # Example predictions with longer contexts
    test_contexts = [
        "the financial affairs committee",
        "in the financial market",
        "the government financial policy",
    ]

    print("\nMaking some predictions:")
    for context in test_contexts:
        words = word_tokenize(context.lower())
        context_tuple = tuple(words)
        predictions = predict_next_word(ngram_model, context_tuple)
        print(f"\nContext: '{context}'")
        print("Top 5 predictions:")
        for word, prob in predictions:
            print(f"  {word}: {prob:.4f}")

    # Generate some sample text with longer seed
    print("\nGenerating sample text:")
    seed_text = "Sometimes I like to "
    generated = generate_text(ngram_model, seed_text, num_words=20)
    print(f"Seed text: '{seed_text}'")
    print(f"Generated: '{generated}'")


if __name__ == "__main__":
    main()
