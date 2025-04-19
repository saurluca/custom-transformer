import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import nltk
import sys
import certifi

sys.path.append('src')

from src.LTSM.lstm import LSTMLanguageModel
from Transformer.transformer import Transformer
from Translation.translator import translate_lstm, translate_decoder_only, translate_encoder_decoder
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
from data.wmt14_dataset.wmt14 import preprocess_wmt14, prepare_dataloader
from Evaluation.bleu_score import calculate_bleu


def test_translation_functions(tokenizer, cfg):
    """
    Test the three translation functions using the WMT14 dataset.
    """
    print("\nTesting translation functions...")

    # Load and preprocess the WMT14 dataset
    print("Preprocessing WMT14 dataset...")
    test_data = preprocess_wmt14("test[:5%]", tokenizer)  # Use 5% of the test split for quick testing
    test_loader = prepare_dataloader(test_data, batch_size=1)

    # Extract example input texts
    example_inputs = []
    for batch in test_loader:
        src, tgt = batch
        example_inputs.append((src.squeeze(0).tolist(), tgt.squeeze(0).tolist()))
        if len(example_inputs) >= 3:  # Limit to 3 examples for testing
            break

    # Decode the example inputs for readability
    example_texts = [tokenizer.decode(src, skip_special_tokens=True) for src, _ in example_inputs]
    print("Example Input Texts:", example_texts)

    # Initialize models
    print("Initializing models...")
    vocab_size = len(tokenizer)
    embedding_dim = cfg.embedding_dim_lstm
    hidden_dim = cfg.hidden_dim_lstm
    num_layers = cfg.num_layers_lstm
    dropout = cfg.dropout_lstm

    # LSTM Model
    lstm_model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(cfg.device)

    # Decoder-Only Transformer Model
    decoder_only_model = Transformer(
        vocab_size, d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_seq_length=cfg.max_seq_length, dropout=cfg.dropout
    ).to(cfg.device)

    # Encoder-Decoder Transformer Model
    encoder_decoder_model = Transformer(
        vocab_size, d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_seq_length=cfg.max_seq_length, dropout=cfg.dropout
    ).to(cfg.device)

    # Test translation functions
    for i, input_text in enumerate(example_texts):
        print(f"\nExample {i + 1}:")
        print(f"Input Text: {input_text}")

        # Translate using LSTM
        translated_lstm = translate_lstm(lstm_model, input_text, tokenizer, cfg.device, max_length=50)
        print(f"LSTM Translation: {translated_lstm}")

        # Translate using Decoder-Only Transformer
        translated_decoder_only = translate_decoder_only(decoder_only_model, input_text, tokenizer, cfg.device, max_length=50)
        print(f"Decoder-Only Transformer Translation: {translated_decoder_only}")

        # Translate using Encoder-Decoder Transformer
        translated_encoder_decoder = translate_encoder_decoder(encoder_decoder_model, input_text, tokenizer, cfg.device, max_length=50)
        print(f"Encoder-Decoder Transformer Translation: {translated_encoder_decoder}")

        print("-" * 50)


def main():
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
    print(f"Vocabulary size: {vocab_size}")
    print("First 10 words in vocabulary:", list(tokenizer.vocab.keys())[:10])

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
        try : 
            torch.save(model.state_dict(), "models/model.pth")
            print("The model has been saved successfully!")
        except Exception as e :
            print(f"failed with the error message: {e}") 
        # save tokenizer
        try :
            torch.save(tokenizer, "models/tokenizer.pth")
            print("The tokenizer has been saved successfully!")
        except Exception as e:
            print(f"failed with the error message: {e}")
        
        # save config
        with open("models/config.json", "w") as f:
            json.dump(cfg, f)
        print("Model, tokenizer, and config saved to models/")

    # Test translation functions
    test_translation_functions(tokenizer, cfg)


if __name__ == "__main__":
    main()
