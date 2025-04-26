import torch


def summarize_lstm(model, input_text, tokenizer, device, max_length=50):
    """
    Generate a summary using an LSTM model.

    Args:
        model: The LSTM model to use for summarization.
        input_text: The input text to summarize.
        tokenizer: A HuggingFace tokenizer instance.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the summarized sequence.
            - A string representing the summarized sequence.
    """
    model.eval()
    with torch.no_grad():
        # Encode input text
        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)

        # Initialize decoder input with start token
        decoder_input = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        generated_tokens = [tokenizer.bos_token_id]

        # Generate tokens one at a time
        for _ in range(max_length):
            # Get model predictions
            outputs, _ = model(decoder_input)
            next_token_logits = outputs[:, -1, :]

            # Get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens.append(next_token)

            # Stop if we predict the end token
            if next_token == tokenizer.eos_token_id:
                break

            # Update decoder input for next iteration
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token]]).to(device)], dim=1
            )

        # Decode the generated tokens
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_tokens, summary


def summarize_decoder_only(model, input_text, tokenizer, device, max_length=50):
    """
    summarize a sequence using a decoder-only Transformer model.

    Args:
        model: The decoder-only Transformer model.
        input_text: The input text to summarize.
        tokenizer: An instance of WordTokenizer or AutoTokenizer.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the summarized sequence.
            - A string representing the summarized sequence.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode the input text into token indices
    input_sequence = tokenizer.encode(input_text)
    input_tensor = (
        torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, seq_len]

    generated_tokens = input_sequence  # Start with the input sequence
    current_input = input_tensor

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass through the model
            output = model(
                current_input, tgt_mask=None
            )  # No mask needed for decoder-only models

            # Get the token with the highest probability (greedy decoding)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated_tokens.append(next_token)

            # Stop if the end-of-sequence token is generated
            if next_token == tokenizer.vocab.get(
                "[SEP]", tokenizer.vocab.get("</s>", None)
            ):
                break

            # Prepare the next input (use the last predicted token)
            current_input = torch.tensor([[next_token]], dtype=torch.long).to(device)

    # Decode the generated tokens back to text
    summarized_text = tokenizer.decode(generated_tokens)
    return generated_tokens, summarized_text


def summarize_encoder_decoder(model, input_text, tokenizer, device, max_length=50):
    """
    Summarize a sequence using an encoder-decoder Transformer model.

    Args:
        model: The encoder-decoder Transformer model.
        input_text: The input text to summarize.
        tokenizer: A HuggingFace tokenizer instance.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the summarized sequence.
            - A string representing the summarized sequence.
    """
    try:
        # Ensure the input is a string
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string")

        if len(input_text.strip()) == 0:
            raise ValueError("Input text cannot be empty")

        model.eval()  # Set the model to evaluation mode

        # Encode the input text and create attention mask
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Create source mask for encoder (1 for non-padding tokens, 0 for padding)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # [batch_size, 1, 1, src_len]

        # Generate the encoder output
        encoder_output = model.encoder(input_ids, src_mask)

        # Initialize the decoder input with the start token
        bos_token_id = (
            tokenizer.bos_token_id
            if hasattr(tokenizer, "bos_token_id")
            else tokenizer.vocab.get("<s>", 1)
        )
        eos_token_id = (
            tokenizer.eos_token_id
            if hasattr(tokenizer, "eos_token_id")
            else tokenizer.vocab.get("</s>", 2)
        )

        decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
        generated_tokens = [bos_token_id]

        with torch.no_grad():
            for i in range(max_length):
                # Create causal mask for decoder (prevent attending to future tokens)
                tgt_mask = (
                    torch.triu(
                        torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                        diagonal=1,
                    )
                    .bool()
                    .to(device)
                )

                # Create cross attention mask
                cross_mask = attention_mask.unsqueeze(1).repeat(
                    1, decoder_input.size(1), 1
                )

                # Generate the decoder output
                output = model.decoder(
                    decoder_input, encoder_output, tgt_mask, cross_mask
                )

                # Get the token with the highest probability (greedy decoding)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                generated_tokens.append(next_token)

                # Stop if the end token is generated
                if next_token == eos_token_id:
                    break

                # Append the next token to the decoder input
                decoder_input = torch.cat(
                    [
                        decoder_input,
                        torch.tensor([[next_token]], dtype=torch.long).to(device),
                    ],
                    dim=1,
                )

        # Decode the generated tokens back to text, skipping special tokens
        summarized_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # If the summary is empty, try a fallback approach
        if not summarized_text.strip():
            print("Empty summary generated, using fallback approach...")
            # Use a simple approach: take the first few sentences of the input
            sentences = input_text.split(".")
            if len(sentences) > 1:
                summarized_text = sentences[0] + ". " + sentences[1] + "."
            else:
                summarized_text = input_text[:100] + "..."

        return generated_tokens, summarized_text

    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        # Return a fallback summary
        return (
            None,
            "This is a fallback summary generated due to an error in the model.",
        )


def summarize(
    model, input_text, tokenizer, device, model_type="encoder_decoder", max_length=50
):
    """
    Summarize a sequence using the specified model type.

    Args:
        model: The model to use for summarization.
        input_text: The input text to summarize.
        tokenizer: A HuggingFace tokenizer instance.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        model_type: The type of model ('lstm', 'decoder_only', or 'encoder_decoder').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the summarized sequence.
            - A string representing the summarized sequence.

    Raises:
        ValueError: If an invalid model type is specified.
    """
    try:
        if not isinstance(input_text, str):
            raise ValueError("input_text must be a string")

        model_type = model_type.lower()
        if model_type == "lstm":
            return summarize_lstm(model, input_text, tokenizer, device, max_length)
        elif model_type == "decoder_only":
            return summarize_decoder_only(
                model, input_text, tokenizer, device, max_length
            )
        elif model_type == "encoder_decoder":
            return summarize_encoder_decoder(
                model, input_text, tokenizer, device, max_length
            )
        else:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be one of: 'lstm', 'decoder_only', 'encoder_decoder'"
            )

    except Exception as e:
        print(f"Error in summarize function: {str(e)}")
        return None, None
