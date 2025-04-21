import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import nltk
import sys
import certifi

sys.path.append('src')

from LTSM.lstm import LSTMLanguageModel
from Transformer.transformer import Transformer
from Translation.translator import translate_lstm, translate_decoder_only, translate_encoder_decoder, translate
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
from Evaluation.bleu_score import calculate_bleu, evaluate_bleu_batch


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



    # Test the model with a few examples
    print("Testing translation...")
    input_texts = [
        "This is a test sentence.",
        "Another example input.",
        "Translate this text."
    ]

    translations = []
    tokenized_translations = []
    for input_text in input_texts:
        # Get both tokenized and decoded translations
        tokenized_translation, translated_text = translate(
            model=model,
            model_type=cfg.model_type,
            input_text=input_text,
            tokenizer=tokenizer,
            device=cfg.device,
            max_length=cfg.seq_length,
        )
        tokenized_translations.append(tokenized_translation)
        translations.append(translated_text)
        print(f"Input: {input_text}")
        print(f"Translated: {translated_text}")
        print("-" * 50)

    # Evaluate BLEU scores for the batch
    tokenized_reference_texts = [tokenizer.encode(text) for text in input_texts]  # Assuming input_texts are references
    bleu_scores, average_bleu = evaluate_bleu_batch(tokenized_translations, tokenized_reference_texts)

    # Print BLEU scores
    for i, score in enumerate(bleu_scores):
        print(f"BLEU Score for input {i + 1}: {score:.4f}")

    print(f"Average BLEU Score: {average_bleu:.4f}")


if __name__ == "__main__":
    main()
