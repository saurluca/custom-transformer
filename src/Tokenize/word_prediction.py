import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import webtext, gutenberg
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Tuple


sys.path.append("sys")

from Transformer.transformer import Transformer, TransformerDecoder
from LSTM.lstm import LSTMLanguageModel
from Summary.summarization import summarize_encoder_decoder


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


def prepare_summarization_sequences(
    examples: List[Tuple[str, str]],  # List of (document, summary) pairs
    tokenizer,
    max_source_length: int = 512,
    max_target_length: int = 128,
):
    """Prepare sequences for summarization training."""
    input_sequences = []
    target_sequences = []

    for document, summary in examples:
        # Tokenize the document
        input_tokens = tokenizer.encode(
            document,
            truncation=True,
            padding="max_length",
            max_length=max_source_length,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Tokenize the summary
        target_tokens = tokenizer.encode(
            summary,
            truncation=True,
            padding="max_length",
            max_length=max_target_length,
            return_tensors="pt",  # Return PyTorch tensors
        )

        input_sequences.append(input_tokens.squeeze(0))  # Remove batch dimension
        target_sequences.append(target_tokens.squeeze(0))  # Remove batch dimension

    return list(zip(input_sequences, target_sequences))


def get_model_class(model_type):
    if model_type == "decoder":
        return TransformerDecoder
    elif model_type == "transformer":
        return Transformer
    elif model_type == "lstm":
        return LSTMLanguageModel
    else:
        raise ValueError(f"Invalid model: {model_type}")


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


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, model_type="lstm"
):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader containing training data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        model_type: Type of model ("lstm", "transformer", or "decoder_only")

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        # Handle different batch formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        elif isinstance(batch, torch.Tensor):
            # For sequence data, split into input and target
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass based on model type
        optimizer.zero_grad()
        
        if model_type == "lstm":
            # LSTM model only takes input and returns output and hidden state
            outputs, _ = model(inputs)
        elif model_type in ["transformer", "decoder_only"]:
            # Create attention masks for transformer models
            src_seq_len = inputs.size(1)
            tgt_seq_len = targets.size(1)

            # Source attention mask (allows attending to all positions)
            src_mask = torch.ones((src_seq_len, src_seq_len), device=device).bool()

            # Target attention mask (causal/triangular mask)
            tgt_mask = ~torch.triu(
                torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1
            ).bool()
            tgt_mask = tgt_mask.to(device)

            # Cross attention mask (allows decoder to attend to all encoder positions)
            if model_type == "transformer":
                cross_mask = torch.ones((tgt_seq_len, src_seq_len), device=device).bool()
                outputs = model(inputs, targets, src_mask, tgt_mask, cross_mask)
            else:  # decoder_only
                outputs = model(inputs, tgt_mask)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Reshape outputs and targets for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        # Calculate loss only on non-padding tokens
        padding_mask = targets != 0
        if padding_mask.any():
            loss = criterion(outputs[padding_mask], targets[padding_mask])
        else:
            loss = torch.tensor(0.0, device=device)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = (
        mask.float()
        .masked_fill(mask == 1, float("-inf"))
        .masked_fill(mask == 0, float(0.0))
    )
    return mask


def evaluate_model(model, test_loader, criterion, device, model_type="lstm"):
    """
    Evaluate the model on the test set.

    Args:
        model: The model to evaluate
        test_loader: DataLoader containing test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        model_type: Type of model ("lstm", "transformer", or "decoder_only")

    Returns:
        float: Average test loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, torch.Tensor):
                # For sequence data, split into input and target
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass based on model type
            if model_type == "lstm":
                # LSTM model only takes input and returns output and hidden state
                outputs, _ = model(inputs)
            elif model_type in ["transformer", "decoder_only"]:
                # Create attention masks for transformer models
                src_seq_len = inputs.size(1)
                tgt_seq_len = targets.size(1)

                # Source attention mask (allows attending to all positions)
                src_mask = torch.ones((src_seq_len, src_seq_len), device=device).bool()

                # Target attention mask (causal/triangular mask)
                tgt_mask = ~torch.triu(
                    torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1
                ).bool()
                tgt_mask = tgt_mask.to(device)

                # Cross attention mask (allows decoder to attend to all encoder positions)
                if model_type == "transformer":
                    cross_mask = torch.ones((tgt_seq_len, src_seq_len), device=device).bool()
                    outputs = model(inputs, targets, src_mask, tgt_mask, cross_mask)
                else:  # decoder_only
                    outputs = model(inputs, tgt_mask)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            # Calculate loss only on non-padding tokens
            padding_mask = targets != 0
            if padding_mask.any():
                loss = criterion(outputs[padding_mask], targets[padding_mask])
            else:
                loss = torch.tensor(0.0, device=device)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


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

    # Determine model type from the model class
    if isinstance(model, LSTMLanguageModel):
        model_type = "lstm"
    elif isinstance(model, Transformer):
        # Check if it's a decoder-only model by looking at the forward method signature
        import inspect
        sig = inspect.signature(model.forward)
        if len(sig.parameters) == 2:  # Only takes input and mask
            model_type = "decoder_only"
        else:
            model_type = "transformer"
    else:
        model_type = cfg.model_type  # Fallback to config

    print(f"Training model type: {model_type}")

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, model_type
        )
        test_loss = evaluate_model(
            model, test_loader, criterion, cfg.device, model_type
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        if not cfg.generate_samples:
            continue

        print("\nGenerating samples:")
        # For summarization, use the first few examples from the test set
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _ = batch
                # Take the first example from the batch
                input_text = tokenizer.decode(
                    inputs[0].tolist(), skip_special_tokens=True
                )
                print(f"Input Text: {input_text}")
                
                # Generate summary using the appropriate model type
                from Summary.summarization import summarize
                _, generated_summary = summarize(
                    model,
                    input_text,
                    tokenizer,
                    cfg.device,
                    model_type=model_type,
                    max_length=cfg.output_length
                )
                print(f"Generated Summary: {generated_summary}")
                print("-" * 50)
                break  # Only process one example

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
    model_type="transformer",
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

            # Get model output based on model type
            if model_type == "lstm":
                output, _ = model(tokens)  # Unpack the tuple for LSTM
            elif model_type == "transformer":
                output = model(tokens, tokens, causal_mask, causal_mask, cross_mask)
            elif model_type == "decoder":
                output = model(tokens, causal_mask)
            else:
                raise ValueError(f"Invalid model: {model_type}")

            next_token_logits = output[:, -1, :] / temperature

            # Set probability of token to 0 to prevent sampling it
            if hasattr(tokenizer, 'use_pretrained') and not tokenizer.use_pretrained:
                unk_idx = tokenizer.vocab["<unk>"]
                next_token_logits[0, unk_idx] = float("-inf")  # Set to negative infinity

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if show_top_k:
                    top_k_values, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                    top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices[0]]
                    print(f"Top {top_k} tokens: {top_k_tokens}")

            
            # Stop if we predict the end token or reach max sequence length
            if hasattr(tokenizer, "use_pretrained") and tokenizer.use_pretrained:
                # For BERT tokenizer, check for [SEP] token
                if (
                    next_token.item() == tokenizer.vocab["[SEP]"]
                    or tokens.size(1) >= seq_length
                ):
                    break
            else:
                # For custom tokenizer, check for end token
                if (
                    next_token.item() == tokenizer.vocab["[END]"]
                    or tokens.size(1) >= seq_length
                ):
                    break

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

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
