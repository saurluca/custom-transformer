import torch

def translate_lstm(model, input_text, tokenizer, device, max_length=50):
    """
    Translate a sequence using the LSTM language model.

    Args:
        model: The LSTM language model.
        input_text: The input text to translate.
        tokenizer: An instance of WordTokenizer.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the translated sequence.
            - A string representing the translated sequence.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode the input text into token indices
    input_sequence = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
    hidden = model.init_hidden(batch_size=1, device=device)

    translated_tokens = []
    current_input = input_tensor

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass through the model
            output, hidden = model(current_input, hidden)
            
            # Get the token with the highest probability (greedy decoding)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            translated_tokens.append(next_token)

            # Stop if the end-of-sequence token is generated
            if next_token == tokenizer.vocab.get("[SEP]", tokenizer.vocab.get("</s>", None)):
                break

            # Prepare the next input (use the last predicted token)
            current_input = torch.tensor([[next_token]], dtype=torch.long).to(device)

    # Decode the translated tokens back to text
    translated_text = tokenizer.decode(translated_tokens)
    return translated_tokens, translated_text


def translate_decoder_only(model, input_text, tokenizer, device, max_length=50):
    """
    Translate a sequence using a decoder-only Transformer model.

    Args:
        model: The decoder-only Transformer model.
        input_text: The input text to translate.
        tokenizer: An instance of WordTokenizer or AutoTokenizer.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the translated sequence.
            - A string representing the translated sequence.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode the input text into token indices
    input_sequence = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    generated_tokens = input_sequence  # Start with the input sequence
    current_input = input_tensor

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass through the model
            output = model(current_input, tgt_mask=None)  # No mask needed for decoder-only models

            # Get the token with the highest probability (greedy decoding)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated_tokens.append(next_token)

            # Stop if the end-of-sequence token is generated
            if next_token == tokenizer.vocab.get("[SEP]", tokenizer.vocab.get("</s>", None)):
                break

            # Prepare the next input (use the last predicted token)
            current_input = torch.tensor([[next_token]], dtype=torch.long).to(device)

    # Decode the generated tokens back to text
    translated_text = tokenizer.decode(generated_tokens)
    return generated_tokens, translated_text


def translate_encoder_decoder(model, input_text, tokenizer, device, max_length=50):
    """
    Translate a sequence using an encoder-decoder Transformer model.

    Args:
        model: The encoder-decoder Transformer model.
        input_text: The input text to translate.
        tokenizer: An instance of WordTokenizer or AutoTokenizer.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length: The maximum length of the generated sequence.

    Returns:
        A tuple containing:
            - A list of token indices representing the translated sequence.
            - A string representing the translated sequence.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode the input text into token indices
    input_sequence = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    # Generate the encoder output
    src_mask = None  # Add a source mask if needed
    encoder_output = model.encoder(input_tensor, src_mask)

    # Initialize the decoder input with the start-of-sequence token
    start_token = tokenizer.vocab.get("<s>", tokenizer.vocab.get("<sos>", None))
    decoder_input = torch.tensor([[start_token]], dtype=torch.long).to(device)

    generated_tokens = [start_token]

    with torch.no_grad():
        for _ in range(max_length):
            # Generate the decoder output
            tgt_mask = None  # Add a target mask if needed
            cross_mask = None  # Add a cross-attention mask if needed
            output = model.decoder(decoder_input, encoder_output, tgt_mask, cross_mask)

            # Get the token with the highest probability (greedy decoding)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated_tokens.append(next_token)

            # Stop if the end-of-sequence token is generated
            if next_token == tokenizer.vocab.get("[SEP]", tokenizer.vocab.get("</s>", None)):
                break

            # Append the next token to the decoder input
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1
            )

    # Decode the generated tokens back to text
    translated_text = tokenizer.decode(generated_tokens)
    return generated_tokens, translated_text


def translate(model, model_type, input_text, tokenizer, device, max_length=50):
    """
    Translate a sequence using the specified model type.

    Args:
        model: The translation model (LSTM, decoder-only Transformer, or encoder-decoder Transformer).
        model_type (str): The type of the model ("lstm", "decoder-only", "encoder-decoder").
        input_text (str): The input text to translate.
        tokenizer: An instance of WordTokenizer or AutoTokenizer.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        max_length (int): The maximum length of the generated sequence.

    Returns:
        str: The translated sequence as a string.
    """
    if model_type == "lstm":
        return translate_lstm(model, input_text, tokenizer, device, max_length)
    elif model_type == "decoder-only":
        return translate_decoder_only(model, input_text, tokenizer, device, max_length)
    elif model_type == "encoder-decoder":
        return translate_encoder_decoder(model, input_text, tokenizer, device, max_length)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")