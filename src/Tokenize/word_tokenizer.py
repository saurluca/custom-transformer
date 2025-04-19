from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import AutoTokenizer

class WordTokenizer:
    def __init__(
        self,
        texts,
        min_freq=2,
        max_vocab_size=10000,
        use_pretrained=False,
        pretrained_model="bert-base-uncased",
    ):
        self.use_pretrained = use_pretrained

        if use_pretrained:
            # Use a pretrained tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
            # Create vocabulary mappings from pretrained tokenizer
            self.vocab = {
                token: idx for token, idx in self.tokenizer.get_vocab().items()
            }
            self.idx2word = {idx: token for token, idx in self.vocab.items()}
        else:
            # Original word-level tokenization
            word_counts = Counter()
            for text in texts:
                word_counts.update(word_tokenize(text.lower()))

            # Create vocabulary with special tokens
            self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}

            # Define punctuation to remove (keeping only , and .)
            punctuation = set('!"#$%&\'()*+-/;<=>?@[\\]^_`{|}~"')
            punctuation.add("--")
            punctuation.add("''")
            punctuation.add("``")
            punctuation.add('""')

            # Add most common words to vocabulary, excluding punctuation
            for word, count in word_counts.most_common(max_vocab_size - 4):
                if count >= min_freq and word not in punctuation:
                    self.vocab[word] = len(self.vocab)

            # Create reverse mapping
            self.idx2word = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        """Convert text to token indices"""
        if self.use_pretrained:
            return self.tokenizer.encode(text).ids
        else:
            tokens = word_tokenize(text.lower())
            return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, indices):
        """Convert token indices back to text"""
        if self.use_pretrained:
            return self.tokenizer.decode(indices)
        else:
            return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])

    def __len__(self):
        return len(self.vocab)

