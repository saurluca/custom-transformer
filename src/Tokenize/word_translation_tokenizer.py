from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import AutoTokenizer

class TranslationTokenizer:
    def __init__(
        self,
        source_texts,  # List of English texts
        target_texts,  # List of German texts
        min_freq=2,
        max_vocab_size=10000,
        use_pretrained=False,
        pretrained_model="bert-base-uncased",  # Consider multilingual models
    ):
        self.use_pretrained = use_pretrained
        self.source_vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.target_vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.source_idx2word = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}
        self.target_idx2word = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}

        if use_pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
            self.source_vocab = self.tokenizer.get_vocab()
            self.source_idx2word = {idx: token for token, idx in self.source_vocab.items()}
            # For target, you might still want a separate vocabulary or use the same if the model is multilingual
            # For simplicity, we'll create a separate one for now.
            target_word_counts = Counter()
            for text in target_texts:
                target_word_counts.update(word_tokenize(text.lower()))

            target_punctuation = set('!"#$%&\'()*+-/;<=>?@[\\]^_`{|}~"')
            target_punctuation.add("--")
            target_punctuation.add("''")
            target_punctuation.add("``")
            target_punctuation.add('""')

            for word, count in target_word_counts.most_common(max_vocab_size - 4):
                if count >= min_freq and word not in target_punctuation:
                    self.target_vocab[word] = len(self.target_vocab)
                    self.target_idx2word[len(self.target_vocab) - 1] = word

        else:
            # Build source vocabulary
            source_word_counts = Counter()
            for text in source_texts:
                source_word_counts.update(word_tokenize(text.lower()))

            source_punctuation = set('!"#$%&\'()*+-/;<=>?@[\\]^_`{|}~"')
            source_punctuation.add("--")
            source_punctuation.add("''")
            source_punctuation.add("``")
            source_punctuation.add('""')

            for word, count in source_word_counts.most_common(max_vocab_size - 4):
                if count >= min_freq and word not in source_punctuation:
                    self.source_vocab[word] = len(self.source_vocab)
                    self.source_idx2word[len(self.source_vocab) - 1] = word

            # Build target vocabulary
            target_word_counts = Counter()
            for text in target_texts:
                target_word_counts.update(word_tokenize(text.lower()))

            target_punctuation = set('!"#$%&\'()*+-/;<=>?@[\\]^_`{|}~"')
            target_punctuation.add("--")
            target_punctuation.add("''")
            target_punctuation.add("``")
            target_punctuation.add('""')

            for word, count in target_word_counts.most_common(max_vocab_size - 4):
                if count >= min_freq and word not in target_punctuation:
                    self.target_vocab[word] = len(self.target_vocab)
                    self.target_idx2word[len(self.target_vocab) - 1] = word

    def encode_source(self, text):
        """Convert source text to token indices"""
        if self.use_pretrained:
            return self.tokenizer.encode(text).ids
        else:
            tokens = word_tokenize(text.lower())
            return [self.source_vocab.get(token, self.source_vocab["<unk>"]) for token in tokens]

    def encode_target(self, text):
        """Convert target text to token indices"""
        tokens = word_tokenize(text.lower())
        return [self.target_vocab.get(token, self.target_vocab["<unk>"]) for token in tokens]

    def decode_source(self, indices):
        """Convert source token indices back to text"""
        if self.use_pretrained:
            return self.tokenizer.decode(indices)
        else:
            return " ".join([self.source_idx2word.get(idx, "<unk>") for idx in indices])

    def decode_target(self, indices):
        """Convert target token indices back to text"""
        return " ".join([self.target_idx2word.get(idx, "<unk>") for idx in indices])

    def __len_source__(self):
        return len(self.source_vocab)

    def __len_target__(self):
        return len(self.target_vocab)