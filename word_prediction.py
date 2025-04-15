import torch
import torch.nn as nn
from nltk.corpus import webtext, gutenberg
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tokenizers import Tokenizer


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


def prepare_sequences(texts, tokenizer, seq_length=10):
    """Prepare sequences for training"""
    sequences = []

    # Process each text as a continuous stream
    for text in texts:
        # Convert text to tokens
        tokens = tokenizer.encode(text)

        # If text is too short, pad it
        if len(tokens) < seq_length + 1:
            # Pad with special tokens to reach required length
            tokens = tokens + [tokenizer.vocab["<pad>"]] * (
                seq_length + 1 - len(tokens)
            )
            # Add this single sequence
            sequences.append(tokens)
        else:
            # Create sliding windows of fixed size
            for i in range(len(tokens) - seq_length):
                seq = tokens[i : i + seq_length + 1]
                sequences.append(seq)

    # Convert to tensor
    return torch.tensor(sequences)


def create_causal_mask(seq_length):
    """Create a causal mask for the transformer"""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return ~mask


def create_cross_attention_mask(tgt_len, src_len):
    """Create a cross attention mask that allows each position in the decoder to attend to all positions in the encoder"""
    # For cross attention, we typically allow attending to all encoder positions
    return torch.ones(tgt_len, src_len).bool()


def init_loss_fn(loss_fn):
    if loss_fn == "NLL":
        return nn.NLLLoss()
    elif loss_fn == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, model_type):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, sequences in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Split sequences into input and target
        inputs = sequences[:, :-1].to(device)
        targets = sequences[:, 1:].to(device)

        # Create masks
        seq_length = inputs.size(1)
        causal_mask = create_causal_mask(seq_length).to(device)
        cross_mask = create_cross_attention_mask(seq_length, seq_length).to(device)

        # Forward pass
        if model_type == "decoder":
            outputs = model(inputs, causal_mask)
        elif model_type == "transformer":
            outputs = model(inputs, inputs, causal_mask, causal_mask, cross_mask)
        else:
            raise ValueError(f"Invalid model: {model_type}")

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


def evaluate_model(model, test_loader, criterion, device, model_type):
    """Evaluate the model on the test set"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, sequences in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Split sequences into input and target
            inputs = sequences[:, :-1].to(device)
            targets = sequences[:, 1:].to(device)

            # Create masks
            seq_length = inputs.size(1)
            causal_mask = create_causal_mask(seq_length).to(device)
            cross_mask = create_cross_attention_mask(seq_length, seq_length).to(device)

            # Forward pass
            if model_type == "decoder":
                outputs = model(inputs, causal_mask)
            elif model_type == "transformer":
                outputs = model(inputs, inputs, causal_mask, causal_mask, cross_mask)
            else:
                raise ValueError(f"Invalid model: {model_type}")

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
    cfg,
):
    train_losses = []
    test_losses = []

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, epoch, cfg.model_type
        )
        test_loss = evaluate_model(
            model, test_loader, criterion, cfg.device, cfg.model_type
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        if not cfg.generate_samples:
            continue

        print("\nGenerating samples:")
        for prompt in cfg.example_prompts:
            # Debug the input tokens
            input_tokens = tokenizer.encode(prompt)

            generated = generate_text(
                model,
                tokenizer,
                prompt,
                output_length=cfg.output_length,
                temperature=cfg.temperature,
                device=cfg.device,
                seq_length=cfg.seq_length_gen,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                sampling_strategy=cfg.sampling_strategy,
                show_top_k=cfg.show_top_k,
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
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
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
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Create a mask for the tokens to keep
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        # Filter out the tokens
        filtered_probs = probs.clone()
        filtered_probs[indices_to_remove] = 0
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        # Sample from the filtered distribution
        return torch.multinomial(filtered_probs, num_samples=1)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")


def generate_text(
    model,
    tokenizer,
    prompt,
    output_length=20,
    temperature=0.7,
    device="cpu",
    seq_length=10,
    show_top_k=False,
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
        for _ in range(output_length):
            # Create masks
            seq_length_current = tokens.size(1)
            causal_mask = create_causal_mask(seq_length_current).to(device)
            cross_mask = create_cross_attention_mask(
                seq_length_current, seq_length_current
            ).to(device)

            # Get model output
            output = model(tokens, tokens, causal_mask, causal_mask, cross_mask)
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


def get_texts(text_names, max_chars_per_text=100000):
    texts = []
    for text_name in text_names:
        if text_name.startswith("gutenberg-"):
            raw_text = gutenberg.raw(text_name[10:])
            # Limit the text length
            if len(raw_text) > max_chars_per_text:
                raw_text = raw_text[:max_chars_per_text]
            texts.append(raw_text)
            pass
        elif text_name.startswith("webtext-"):
            # Just get the raw text without any sentence splitting
            raw_text = webtext.raw(text_name[8:])
            # Limit the text length
            if len(raw_text) > max_chars_per_text:
                raw_text = raw_text[:max_chars_per_text]
            texts.append(raw_text)
    return texts


def split_train_test(sequences, train_size=0.9):
    train_size = int(train_size * len(sequences))
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    return train_sequences, test_sequences
