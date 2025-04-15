import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from word_prediction import (
    WordTokenizer,
    get_texts,
    prepare_sequences,
    init_loss_fn,
    train_model,
    plot_loss,
    split_train_test,
)
from transformer import TransformerDecoder, Transformer
from config import cfg
import nltk


# Download necessary NLTK data
print("Downloading NLTK data...")
nltk.download("gutenberg")
nltk.download("webtext")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")


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

    print("Initializing model...")
    print(f"Using device: {cfg.device}")
    
    if cfg.model_type == "decoder":
        model = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_seq_length=cfg.max_seq_length,
        ).to(cfg.device)
    elif cfg.model_type == "transformer":
        model = Transformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_seq_length=cfg.max_seq_length,
        ).to(cfg.device)
    else:
        raise ValueError(f"Invalid model: {cfg.model_type}")

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
        torch.save(model.state_dict(), "models/model.pth")
        # save tokenizer
        torch.save(tokenizer, "models/tokenizer.pth")
        # save config
        with open("models/config.json", "w") as f:
            json.dump(cfg, f)
        print("Model, tokenizer, and config saved to models/")


if __name__ == "__main__":
    main()
