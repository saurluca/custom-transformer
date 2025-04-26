import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# from dotenv import load_dotenv
from transformers import AutoTokenizer

# Add src directory to Python path
sys.path.append('src')

from LSTM.lstm import LSTMLanguageModel
from Transformer.transformer import Transformer
from Summary.summarization import (
    summarize,
)
from Translation.translator import (
    translate_lstm,
    translate_decoder_only,
    translate_encoder_decoder,
)
from Tokenize.word_prediction import (
    get_texts,
    prepare_sequences,
    init_loss_fn,
    train_model,
    plot_loss,
    split_train_test,
    get_model_class,
)
from Config.config import cfg
from Tokenize.word_tokenizer import WordTokenizer
from data.xl_sum_dataset.xl_sum import (
    prepare_dataloader_xl_sum,
    load_or_preprocess_xl_sum_data,
)
from data.wmt14_dataset.wmt14 import preprocess_wmt14, prepare_dataloader
from Evaluation.rouge_score import calculate_rouge_scores, plot_rouge_scores

# load_dotenv()
# if hf_token:=os.getenv("HUGGINGFACE_TOKEN"):
#     login(token=hf_token)
#     print("Your HF account has been successfully logged in!")


def Translator():
    """
    Test the three translation functions using the WMT14 dataset.
    """
    print("\nTesting translation functions...")

    # Load and preprocess the WMT14 dataset
    print("Preprocessing WMT14 dataset...")
    tokenizer_name = cfg.translation_tokenizer
    print(f"name of the tokenizer is : {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    test_data = preprocess_wmt14(
        "test[:5%]", tokenizer
    )  # Use 5% of the test split for quick testing
    test_loader = prepare_dataloader(test_data, batch_size=32)

    # Extract example input texts
    example_inputs = []
    for batch in test_loader:
        src, tgt = batch
        example_inputs.append((src.squeeze(0).tolist(), tgt.squeeze(0).tolist()))
        if len(example_inputs) >= 3:
            break

    print("\n Content of example_inputs: ")
    for src_list, _ in example_inputs:
        print(f"Source Token list: {src_list}, Type:{type(src_list)}")
    # Decode the example inputs for readability
    example_texts = [
        tokenizer.decode(src_list, skip_special_tokens=True)
        for src_list, _ in example_inputs
    ]
    print("Example Input Texts:", example_texts)

    # Initialize models
    print("Initializing models...")
    vocab_size = len(tokenizer)
    embedding_dim = cfg.embedding_dim_lstm
    hidden_dim = cfg.hidden_dim_lstm
    num_layers = cfg.num_layers_lstm
    dropout = cfg.dropout_lstm

    # LSTM Model
    lstm_model = LSTMLanguageModel(
        vocab_size, embedding_dim, hidden_dim, num_layers, dropout
    ).to(cfg.device)

    # Decoder-Only Transformer Model
    decoder_only_model = Transformer(
        vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # Encoder-Decoder Transformer Model
    encoder_decoder_model = Transformer(
        vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # Test translation functions
    for i, input_text in enumerate(example_texts):
        print(f"\nExample {i + 1}:")
        print(f"Input Text: {input_text}")

        # Translate using LSTM
        translated_lstm = translate_lstm(
            lstm_model, input_text, tokenizer, cfg.device, max_length=50
        )
        print(f"LSTM Translation: {translated_lstm}")

        # Translate using Decoder-Only Transformer
        translated_decoder_only = translate_decoder_only(
            decoder_only_model, input_text, tokenizer, cfg.device, max_length=50
        )
        print(f"Decoder-Only Transformer Translation: {translated_decoder_only}")

        # Translate using Encoder-Decoder Transformer
        translated_encoder_decoder = translate_encoder_decoder(
            encoder_decoder_model, input_text, tokenizer, cfg.device, max_length=50
        )
        print(f"Encoder-Decoder Transformer Translation: {translated_encoder_decoder}")

    if cfg.save_model:
        print("Saving model...")
    # save model
    try:
        torch.save(model.state_dict(), "models/model_translation.pth")
        print("The model has been saved successfully!")
    except Exception as e:
        print(f"failed with the error message: {e}")
    # save tokenizer
    try:
        torch.save(tokenizer, "models/tokenizer_translation.pth")
        print("The tokenizer has been saved successfully!")
    except Exception as e:
        print(f"failed with the error message: {e}")

    # save config
    with open("models/config_translation.json", "w") as f:
        json.dump(cfg, f)
    print("Model, tokenizer, and config saved to models")

    print("-" * 50)


def Summarization():
    print("Summarization mode")
    """
    Train and evaluate summarization model using the XL-sum dataset.
    """
    print("\nStarting summarization task...")

    # Load the tokenizer
    print(f"Loading tokenizer: {cfg.summarization_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.summarization_tokenizer)

    # Define dataset directory
    dataset_dir = "data/xl_sum_dataset/preprocessed"
    os.makedirs(dataset_dir, exist_ok=True)

    # Load or preprocess the dataset - using a much smaller subset for testing
    print("Loading or preprocessing XL-sum dataset (small subset for testing)...")
    train_data = load_or_preprocess_xl_sum_data(
        "train[:1%]", tokenizer, dataset_dir, max_length=cfg.max_seq_length
    )
    test_data = load_or_preprocess_xl_sum_data(
        "test[:1%]", tokenizer, dataset_dir, max_length=cfg.max_seq_length
    )

    # Prepare dataloaders
    print("Preparing dataloaders...")
    train_loader = prepare_dataloader_xl_sum(train_data, batch_size=cfg.batch_size, max_length=cfg.max_seq_length)
    test_loader = prepare_dataloader_xl_sum(test_data, batch_size=cfg.batch_size, max_length=cfg.max_seq_length)
    
    # Initialize models
    print("Initializing models...")
    vocab_size = len(tokenizer)
    
    # LSTM Model
    lstm_model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim_lstm,
        hidden_dim=cfg.hidden_dim_lstm,
        num_layers=cfg.num_layers_lstm,
        dropout=cfg.dropout_lstm,
    ).to(cfg.device)
    
    # Decoder-Only Transformer Model
    decoder_only_model = Transformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout
    ).to(cfg.device)
    
    # Encoder-Decoder Transformer Model
    encoder_decoder_model = Transformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # Initialize loss function and optimizer
    criterion = init_loss_fn(cfg.loss_fn)
    
    # Train models
    print("Training models...")
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lstm_train_losses, lstm_test_losses = train_model(
        lstm_model, train_loader, test_loader, tokenizer, lstm_optimizer, criterion, cfg
    )
    
    # Train Decoder-Only model
    print("\nTraining Decoder-Only model...")
    decoder_optimizer = optim.Adam(decoder_only_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    decoder_train_losses, decoder_test_losses = train_model(
        decoder_only_model, train_loader, test_loader, tokenizer, decoder_optimizer, criterion, cfg
    )
    
    # Train Encoder-Decoder model
    print("\nTraining Encoder-Decoder model...")
    encoder_decoder_optimizer = optim.Adam(encoder_decoder_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    encoder_decoder_train_losses, encoder_decoder_test_losses = train_model(
        encoder_decoder_model, train_loader, test_loader, tokenizer, encoder_decoder_optimizer, criterion, cfg
    )

    # Plot loss curves
    print("Plotting loss curves...")
    plot_loss(lstm_train_losses, lstm_test_losses, save_path="plots/lstm_summarization_loss.png")
    plot_loss(decoder_train_losses, decoder_test_losses, save_path="plots/decoder_only_summarization_loss.png")
    plot_loss(encoder_decoder_train_losses, encoder_decoder_test_losses, save_path="plots/encoder_decoder_summarization_loss.png")
    
    # Test summarization on example texts
    print("\nTesting summarization on example texts...")
    example_texts = [
        "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet at least once. Pangrams have been used to display typefaces and test equipment since the invention of printing.",
        "Artificial intelligence has transformed various sectors including healthcare, finance, and transportation. Machine learning algorithms can now diagnose diseases, predict market trends, and drive autonomous vehicles. However, concerns about AI ethics and safety continue to grow.",
        "Climate change poses significant challenges to global ecosystems. Rising temperatures, extreme weather events, and sea level rise threaten both human communities and wildlife. Scientists emphasize the need for immediate action to reduce greenhouse gas emissions."
    ]
    
    # Get ground truth summaries from the test set
    print("\nGetting ground truth summaries from test set...")
    ground_truth_summaries = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            src, tgt = batch
        else:
            src = batch['input_ids']
            tgt = batch['labels']
            
        # Convert tensors to lists and filter out padding tokens (0)
        src_tokens = [token for token in src[0].tolist() if token != 0]
        tgt_tokens = [token for token in tgt[0].tolist() if token != 0]
        
        # Convert lists to tensors for decoding
        src_tensor = torch.tensor(src_tokens)
        tgt_tensor = torch.tensor(tgt_tokens)
        
        # Decode the tokens to text
        src_text = tokenizer.decode(src_tensor, skip_special_tokens=True)
        tgt_text = tokenizer.decode(tgt_tensor, skip_special_tokens=True)
        
        ground_truth_summaries.append(tgt_text)
        
        if len(ground_truth_summaries) >= len(example_texts):
            break
    
    # Generate summaries for each model type and calculate ROUGE scores
    model_types = ["lstm", "decoder_only", "encoder_decoder"]
    models = {
        "lstm": lstm_model,
        "decoder_only": decoder_only_model,
        "encoder_decoder": encoder_decoder_model
    }
    
    all_rouge_scores = {}
    
    for model_type in model_types:
        print(f"\nGenerating summaries using {model_type} model...")
        generated_summaries = []
        
        for i, text in enumerate(example_texts):
            print(f"\nExample {i + 1}:")
            print(f"Input text: {text[:100]}...")
            
            # Generate summary using the current model
            _, summary = summarize(models[model_type], text, tokenizer, cfg.device, model_type=model_type)
            
            if summary:
                print(f"Generated summary: {summary}")
                generated_summaries.append(summary)
            else:
                print("Failed to generate summary")
                generated_summaries.append("")  # Add empty string for failed summaries
        
        # Calculate and plot ROUGE scores for this model
        print(f"\nCalculating ROUGE scores for {model_type} model...")
        rouge_scores = calculate_rouge_scores(generated_summaries, ground_truth_summaries)
        all_rouge_scores[model_type] = rouge_scores
        
        print(f"\nROUGE Scores for {model_type} model:")
        for metric, score in rouge_scores.items():
            print(f"{metric}: {score:.4f}")
        
        # Plot ROUGE scores for this model
        plot_rouge_scores(rouge_scores, save_path=f"plots/rouge_scores_{model_type}.png")
    
    # Plot comparison of ROUGE scores across models
    print("\nPlotting comparison of ROUGE scores across models...")
    # This would require modifying the plot_rouge_scores function to handle multiple models
    # For now, we'll just print the comparison
    print("\nROUGE Score Comparison:")
    print("Model Type | ROUGE-1 | ROUGE-2 | ROUGE-L")
    print("-" * 40)
    for model_type, scores in all_rouge_scores.items():
        print(f"{model_type:12} | {scores['rouge1']:.4f} | {scores['rouge2']:.4f} | {scores['rougeL']:.4f}")
    
    # Save models if configured
    if cfg.save_model:
        print("\nSaving models...")
        try:
            torch.save(lstm_model.state_dict(), "models/lstm_summarization.pth")
            torch.save(decoder_only_model.state_dict(), "models/decoder_only_summarization.pth")
            torch.save(encoder_decoder_model.state_dict(), "models/encoder_decoder_summarization.pth")
            print("Models saved successfully!")
        except Exception as e:
            print(f"Failed to save models with error: {e}")
        
        # Save tokenizer
        try:
            tokenizer.save_pretrained("models/summarization_tokenizer")
            print("Tokenizer saved successfully!")
        except Exception as e:
            print(f"Failed to save tokenizer with error: {e}")

        # Save config
        with open("models/config_summarization.json", "w") as f:
            json.dump(cfg.__dict__, f)
        print("Config saved successfully!")


def next_word_generator():
    print("Loading texts...")
    texts = get_texts(cfg.text_names, cfg.max_chars_per_text)
    print(f"Total samples: {len(texts)}")

    print("Initializing tokenizer...")
    tokenizer = WordTokenizer(
        texts,
        min_freq=cfg.min_vocab_freq,
        max_vocab_size=cfg.max_vocab_size,
        use_pretrained=cfg.use_pretrained,
        pretrained_model=cfg.pretrained_model,
    )
    vocab_size = len(tokenizer)
    # print(f"Vocabulary size: {vocab_size}")
    # print("First 10 words in vocabulary:", list(tokenizer.vocab.keys())[:10])

    print("Preparing sequences...")
    sequences = prepare_sequences(texts, tokenizer, seq_length=cfg.seq_length)

    print("Splitting into train and test sets...")
    train_sequences, test_sequences = split_train_test(sequences)

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_sequences,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_sequences, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    print(f"Initializing {cfg.model_type} model...")
    print(f"Using device: {cfg.device}")

    if cfg.model_type == "lstm":
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=cfg.embedding_dim_lstm,
            hidden_dim=cfg.hidden_dim_lstm,
            num_layers=cfg.num_layers_lstm,
            dropout=cfg.dropout_lstm,
        ).to(cfg.device)
    else:
        modelClass = get_model_class(cfg.model_type)
        model = modelClass(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_seq_length=cfg.max_seq_length,
        ).to(cfg.device)

    print(f"Model parameters number: {sum(p.numel() for p in model.parameters())}")

    print("Initializing loss and optimizer...")
    criterion = init_loss_fn(cfg.loss_fn)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    print("Starting training...")
    train_losses, test_losses = train_model(
        model,
        train_loader,
        test_loader,
        tokenizer,
        optimizer,
        criterion,
        cfg,
    )
    print("Training completed!")

    print("Plotting loss curves...")
    plot_loss(train_losses, test_losses)

    if cfg.save_model:
        print("Saving model...")
        # save model
        try:
            torch.save(model.state_dict(), "models/model_next_word_generator.pth")
            print("The model has been saved successfully!")
        except Exception as e:
            print(f"failed with the error message: {e}")
        # save tokenizer
        try:
            torch.save(tokenizer, "models/tokenizer_next_word_generator.pth")
            print("The tokenizer has been saved successfully!")
        except Exception as e:
            print(f"failed with the error message: {e}")

        # save config
        with open("models/config_next_word_generator.json", "w") as f:
            json.dump(cfg, f)
        print("Model, tokenizer, and config saved to models")


def evaluate_reconstruction(
    model,
    tokenizer,
    text,
    mask_ratio=0.3,
    device="cpu",
    num_sequences=100,
):
    """
    Evaluate how well the model can reconstruct a text by masking part of the sequences
    and measuring the model's predictions against the original tokens.

    Args:
        model: Trained model
        tokenizer: Tokenizer used to convert text to sequences
        text: Text to evaluate on
        mask_ratio: Proportion of sequence to mask for prediction
        device: Device to run the model on

    Returns:
        Dictionary with ROUGE and BLEU scores
    """
    # only use the percnetage of text_ratio of the text
    print(f"Evaluating reconstruction on text of length {len(text)}")
    model.eval()

    # filter for the text_len
    print("Creating sequences...")
    # Create sequences from the text
    sequences = prepare_sequences([text], tokenizer, seq_length=cfg.seq_length)
    sequences = sequences[:num_sequences]
    print("Initializing scorers...")
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    # Store scores
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    bleu_scores = []

    print("Evaluating...")
    with torch.no_grad():
        for sequence in tqdm(sequences):
            input_seq = sequence[:-1].unsqueeze(0).to(device)
            target_seq = sequence[1:].unsqueeze(0).to(device)

            # Determine how many tokens to mask
            seq_len = input_seq.size(1)
            mask_len = int(seq_len * mask_ratio)
            visible_len = seq_len - mask_len

            # Keep only the visible part of the sequence
            partial_input = input_seq[:, :visible_len]

            # Generate the entire sequence from the partial input
            if cfg.model_type == "lstm":
                outputs, _ = model(partial_input)
                # Get the predictions for the masked part
                predictions = outputs[:, -1, :]

                # Generate completions token by token
                generated_tokens = []
                curr_input = partial_input.clone()

                for _ in range(mask_len):
                    with torch.no_grad():
                        output, _ = model(curr_input)
                        prediction = output[:, -1, :]
                        next_token = prediction.argmax(dim=-1).unsqueeze(0)
                        generated_tokens.append(next_token.item())
                        curr_input = torch.cat([curr_input, next_token], dim=1)
            else:
                # For transformer models
                causal_mask = create_causal_mask(visible_len).to(device)
                cross_mask = create_cross_attention_mask(visible_len, visible_len).to(
                    device
                )

                if cfg.model_type == "decoder":
                    outputs = model(partial_input, causal_mask)
                else:  # transformer
                    outputs = model(
                        partial_input,
                        partial_input,
                        causal_mask,
                        causal_mask,
                        cross_mask,
                    )

                # Generate completions token by token
                generated_tokens = []
                curr_input = partial_input.clone()

                for _ in range(mask_len):
                    with torch.no_grad():
                        curr_len = curr_input.size(1)
                        causal_mask = create_causal_mask(curr_len).to(device)
                        cross_mask = create_cross_attention_mask(curr_len, curr_len).to(
                            device
                        )

                        if cfg.model_type == "decoder":
                            output = model(curr_input, causal_mask)
                        else:  # transformer
                            output = model(
                                curr_input,
                                curr_input,
                                causal_mask,
                                causal_mask,
                                cross_mask,
                            )

                        prediction = output[:, -1, :]
                        next_token = prediction.argmax(dim=-1).unsqueeze(1)
                        generated_tokens.append(next_token.item())
                        curr_input = torch.cat([curr_input, next_token], dim=1)

            # Get the target tokens (what should have been generated)
            target_tokens = target_seq[:, visible_len - 1 :].squeeze().cpu().tolist()

            # Convert tokens to text
            predicted_text = tokenizer.decode(generated_tokens)
            target_text = tokenizer.decode(target_tokens)

            # Calculate ROUGE scores
            scores = rouge.score(target_text, predicted_text)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

            # Calculate BLEU score
            predicted_tokens = [
                tokenizer.idx2word.get(idx, "<unk>") for idx in generated_tokens
            ]
            target_tokens = [
                tokenizer.idx2word.get(idx, "<unk>") for idx in target_tokens
            ]
            bleu = sentence_bleu(
                [target_tokens], predicted_tokens, smoothing_function=smoother
            )
            bleu_scores.append(bleu)

    print("Averaging scores...")
    # Average scores
    avg_rouge1 = (
        sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
        if rouge_scores["rouge1"]
        else 0
    )
    avg_rouge2 = (
        sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
        if rouge_scores["rouge2"]
        else 0
    )
    avg_rougeL = (
        sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])
        if rouge_scores["rougeL"]
        else 0
    )
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    results = {
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "bleu": avg_bleu,
    }

    print("Evaluation Results:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"BLEU: {avg_bleu:.4f}")

    return results


def plot_model_comparison(results, param_counts):
    """
    Plot model comparison results with ROUGE and BLEU scores.

    Args:
        results: Dictionary with model names as keys, each containing score dictionaries
        param_counts: Dictionary with model names as keys and parameter counts as values
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prepare data for plotting
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.2

    # Plot ROUGE scores
    rouge1_scores = [results[model]["rouge1"] for model in models]
    rouge2_scores = [results[model]["rouge2"] for model in models]
    rougeL_scores = [results[model]["rougeL"] for model in models]

    ax1.bar(x - width, rouge1_scores, width, label="ROUGE-1")
    ax1.bar(x, rouge2_scores, width, label="ROUGE-2")
    ax1.bar(x + width, rougeL_scores, width, label="ROUGE-L")

    # Add model parameter counts on x-axis labels
    model_labels = [
        f"{model}\n({param_counts[model] / 1e6:.2f}M params)" for model in models
    ]

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Score")
    ax1.set_title("ROUGE Scores Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot BLEU scores
    bleu_scores = [results[model]["bleu"] for model in models]

    ax2.bar(x, bleu_scores, width * 2, label="BLEU")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Score")
    ax2.set_title("Model Parameter Counts")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("Comparison plot saved to 'model_comparison.png'")
    plt.close()


def compare_all_models(texts, tokenizer):
    """
    Train and evaluate all three models (LSTM, Decoder, Transformer)
    and compare their reconstruction capabilities.

    Args:
        texts: List of texts for training and evaluation
        tokenizer: Tokenizer for processing texts
    """
    print("Starting model comparison...")
    vocab_size = len(tokenizer)

    # Store original model type
    original_model_type = cfg.model_type

    # Prepare sequences
    sequences = prepare_sequences(texts, tokenizer, seq_length=cfg.seq_length)
    train_sequences, test_sequences = split_train_test(sequences)

    # Create data loaders
    train_loader = DataLoader(
        train_sequences,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_sequences, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # Models to compare
    model_types = ["lstm", "decoder", "transformer"]
    results = {}
    param_counts = {}

    # Train and evaluate each model
    for model_type in model_types:
        print(f"\n{'-' * 50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'-' * 50}")

        # Set current model type
        cfg.model_type = model_type

        # Initialize model
        if model_type == "lstm":
            model = LSTMLanguageModel(
                vocab_size=vocab_size,
                embedding_dim=cfg.embedding_dim_lstm,
                hidden_dim=cfg.hidden_dim_lstm,
                num_layers=cfg.num_layers_lstm,
                dropout=cfg.dropout_lstm,
            ).to(cfg.device)
        else:
            modelClass = get_model_class(model_type)
            if model_type == "transformer":
                num_layers = cfg.num_layers // 2
            else:
                num_layers = cfg.num_layers
            model = modelClass(
                vocab_size=vocab_size,
                d_model=cfg.d_model,
                num_layers=num_layers,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                max_seq_length=cfg.max_seq_length,
            ).to(cfg.device)

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_counts[model_type] = param_count
        print(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

        # Initialize loss and optimizer
        criterion = init_loss_fn(cfg.loss_fn)
        if model_type == "lstm":
            optimizer = optim.Adam(
                model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.weight_decay
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )

        # Train model
        train_losses, test_losses = train_model(
            model,
            train_loader,
            test_loader,
            tokenizer,
            optimizer,
            criterion,
            cfg,
        )

        # Evaluate model
        print(f"Evaluating {model_type.upper()} model reconstruction ability...")
        if len(texts) > 0:
            reconstruction_scores = evaluate_reconstruction(
                model,
                tokenizer,
                texts[0],
                mask_ratio=cfg.mask_ratio,
                device=cfg.device,
            )
            results[model_type] = reconstruction_scores

    # Restore original model type
    cfg.model_type = original_model_type

    # Plot comparison results
    plot_model_comparison(results, param_counts)

    # Print parameter comparison
    print("\nModel parameter counts:")
    for model_type, count in param_counts.items():
        print(f"{model_type.upper()}: {count:,} parameters ({count / 1e6:.2f}M)")

    return results, param_counts



if __name__ == "__main__":
    if cfg.mode == "next-word-generation":
        next_word_generator()
    elif cfg.mode == "summarization":
        Summarization()
    elif cfg.mode == "translation":
        Translator()
    else:
        print(
            f"Unknown mode: {cfg.mode}. Available modes: next-word-generation, summarization, translation"
        )
