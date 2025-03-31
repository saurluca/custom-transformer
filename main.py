import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def download_nltk_data():
    nltk.download("brown")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")


class CustomTransformer:
    def __init__(self):
        pass

    def predict(self, data: dict):
        return data


def load_data():
    return brown.sents()


def preprocess_data(data: list):
    # concatenate all sentences into a single string
    text = " ".join([" ".join(sentence) for sentence in data])

    # tokenize the text into words
    words = word_tokenize(text.lower())

    # remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    return words


def build_vocab(data: list):
    # data is now a list of words, so we can directly create a set from it
    return set(data)


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


if __name__ == "__main__":
    main()
