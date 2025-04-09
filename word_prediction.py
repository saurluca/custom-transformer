import json
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from collections import Counter
from transformer import TransformerDecoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tokenizers import Tokenizer

# Download necessary NLTK data
print("Downloading NLTK data...")
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")


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
            self.tokenizer = Tokenizer.from_pretrained(pretrained_model)
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
            punctuation = set('!"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~"')

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


def prepare_sequences(texts, tokenizer, seq_length=10):
    """Prepare sequences for training"""
    sequences = []
    for text in texts:
        tokens = tokenizer.encode(text)
        # Create sequences of length seq_length + 1 (input + target)
        for i in range(len(tokens) - seq_length):
            seq = tokens[i : i + seq_length + 1]
            sequences.append(seq)
    return torch.tensor(sequences)


def create_causal_mask(seq_length):
    """Create a causal mask for the transformer"""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return ~mask


def init_loss_fn(loss_fn):
    if loss_fn == "NLL":
        return nn.NLLLoss()
    elif loss_fn == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, sequences in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Split sequences into input and target
        inputs = sequences[:, :-1].to(device)
        targets = sequences[:, 1:].to(device)

        # Create causal mask
        seq_length = inputs.size(1)
        mask = create_causal_mask(seq_length).to(device)

        # Forward pass
        outputs = model(inputs, mask)

        # Reshape outputs and targets for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, sequences in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Split sequences into input and target
            inputs = sequences[:, :-1].to(device)
            targets = sequences[:, 1:].to(device)

            # Create causal mask
            seq_length = inputs.size(1)
            mask = create_causal_mask(seq_length).to(device)

            # Forward pass
            outputs = model(inputs, mask)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_model(
    model,
    train_loader,
    test_loader,
    tokenizer,
    optimizer,
    criterion,
    device,
    cfg,
):
    # Training loop
    print("Starting training...")
    train_losses = []
    test_losses = []

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss = evaluate_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        print("\nGenerating samples:")
        for prompt in cfg.example_prompts:
            # Debug the input tokens
            input_tokens = tokenizer.encode(prompt)

            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=cfg.max_length_gen,
                temperature=cfg.temperature,
                device=device,
                seq_length=cfg.seq_length_gen,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                sampling_strategy=cfg.sampling_strategy,
            )
            print(f"Prompt: {prompt}")
            print(f"Decoded input: {tokenizer.decode(input_tokens)}")
            print(f"Generated: {generated}")
            print("-" * 50)

    return train_losses, test_losses


def sample_next_token(probs, sampling_strategy, top_k=5, top_p=0.8):
    if sampling_strategy == "multinomial":
        return torch.multinomial(probs, num_samples=1)
    elif sampling_strategy == "greedy":
        return torch.argmax(probs, dim=-1).unsqueeze(0)
    elif sampling_strategy == "top-k":
        # Get top k values and indices
        top_k_values, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        # Resample probabilities to only include top k
        filtered_probs = torch.zeros_like(probs).scatter_(
            -1, top_k_indices, top_k_values
        )
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(
            dim=-1, keepdim=True
        )
        # Sample from the filtered distribution
        return torch.multinomial(filtered_probs, num_samples=1)
    elif sampling_strategy == "top-p":
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Create a mask for the tokens to keep
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        # Filter out the tokens
        filtered_probs = probs.clone()
        filtered_probs[indices_to_remove] = 0
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(
            dim=-1, keepdim=True
        )
        # Sample from the filtered distribution
        return torch.multinomial(filtered_probs, num_samples=1)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
    
def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=20,
    temperature=0.7,
    device="cpu",
    seq_length=10,
    show_top_k=True,
    top_k=5,
    top_p=0.8,
    sampling_strategy="multinomial",
):
    """Generate text from a prompt"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    # Ensure we don't exceed max_seq_length
    tokens = tokens[: seq_length - 1]  # Leave room for at least one new token
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Create causal mask
            seq_length_current = tokens.size(1)
            mask = create_causal_mask(seq_length_current).to(device)

            # Get model output
            output = model(tokens, mask)
            next_token_logits = output[:, -1, :] / temperature

            # Set probability of <unk> token to 0 to prevent sampling it
            if not tokenizer.use_pretrained:
                unk_idx = tokenizer.vocab["<unk>"]
                next_token_logits[0, unk_idx] = float(
                    "-inf"
                )  # Set to negative infinity

            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = sample_next_token(probs, sampling_strategy, top_k, top_p)

            if show_top_k:
                # print the input sentence
                print(f"Input: {tokenizer.decode(tokens[0].tolist())}")
                # Get top k most likely next tokens
                top_probs, top_indices = torch.topk(probs[0], top_k)
                print("\nTop 5 most likely next words:")
                for prob, idx in zip(top_probs, top_indices):
                    word = tokenizer.idx2word[idx.item()]
                    print(f"  {word}: {prob.item():.4f}")

                # Print the selected word and its probability
                selected_word = tokenizer.idx2word[next_token.item()]
                selected_prob = probs[0, next_token.item()].item()
                print(
                    f"\nSelected word: {selected_word} (probability: {selected_prob:.4f})"
                )
                print()

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if we predict the end token or reach max sequence length
            if tokenizer.use_pretrained:
                # For BERT tokenizer, check for [SEP] token
                if (
                    next_token.item() == tokenizer.vocab["[SEP]"]
                    or tokens.size(1) >= seq_length
                ):
                    break
            else:
                # For custom tokenizer, check for </s> token
                if (
                    next_token.item() == tokenizer.vocab["</s>"]
                    or tokens.size(1) >= seq_length
                ):
                    break

    return tokenizer.decode(tokens[0].tolist())


def plot_loss(train_losses, test_losses, save_path="plots/loss.png"):
    """Plot training and test loss over epochs and save to file."""
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()


def main():
    cfg = SimpleNamespace(**{})

    # run config
    cfg.save_model = False

    # data
    cfg.dataset = "austen-emma.txt"
    cfg.num_samples = 1000
    cfg.min_vocab_freq = 2
    cfg.max_vocab_size = 10000
    cfg.seq_length = 10
    cfg.train_size = 0.9
    cfg.use_pretrained = False
    cfg.pretrained_model = "bert-base-uncased"

    # training
    cfg.batch_size = 128
    cfg.num_epochs = 1
    cfg.learning_rate = 0.001
    cfg.weight_decay = 0.0001
    cfg.loss_fn = "CrossEntropyLoss"  # "CrossEntropyLoss", "NLL"

    # model
    cfg.d_model = 256
    cfg.num_layers = 4
    cfg.num_heads = 4
    cfg.d_ff = 1024
    cfg.dropout = 0.1
    cfg.max_seq_length = 20
    cfg.max_length = 15

    # text generation
    cfg.max_length_gen = 15  # max length of generated text
    cfg.seq_length_gen = 10  # sequence length for generation
    cfg.temperature = 0.7
    cfg.top_k = 5
    cfg.top_p = 0.5
    cfg.sampling_strategy = "top-k"  # "multinomial", "greedy", "top-k", "top-p"
    cfg.example_prompts = ["the man who", "her mother had", "she was the", "I love "]

    # Load a small dataset from NLTK Gutenberg
    print("Loading dataset...")
    texts = gutenberg.raw(cfg.dataset).split(".")[: cfg.num_samples]

    # Initialize tokenizer with larger vocabulary and lower frequency threshold
    print("Initializing tokenizer...")
    tokenizer = WordTokenizer(
        texts,
        min_freq=cfg.min_vocab_freq,
        max_vocab_size=cfg.max_vocab_size,
        use_pretrained=cfg.use_pretrained,
        pretrained_model=cfg.pretrained_model,
    )
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # print the first 10 words
    print("First 10 words in vocabulary:", list(tokenizer.vocab.keys())[:10])

    # Prepare sequences
    print("Preparing sequences...")
    sequences = prepare_sequences(texts, tokenizer, seq_length=cfg.seq_length)

    # Split into train and test sets
    train_size = int(cfg.train_size * len(sequences))
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    print(f"Train size: {len(train_sequences)}, Test size: {len(test_sequences)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_sequences,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(test_sequences, batch_size=cfg.batch_size)

    # Initialize model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_length=cfg.max_seq_length,
    ).to(device)

    # Loss and optimizer
    criterion = init_loss_fn(cfg.loss_fn)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    train_losses, test_losses = train_model(
        model,
        train_loader,
        test_loader,
        tokenizer,
        optimizer,
        criterion,
        device,
        cfg,
    )

    # Plot and save the loss curves
    plot_loss(train_losses, test_losses)

    print("Training completed!")

    if cfg.save_model:
        # save model
        torch.save(model.state_dict(), "models/model.pth")

        # save tokenizer
        torch.save(tokenizer, "models/tokenizer.pth")

        # save config
        with open("models/config.json", "w") as f:
            json.dump(cfg, f)

        print("Model, tokenizer, and config saved to models/")


if __name__ == "__main__":
    main()
