import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import nltk
import sys
import os
import certifi
from huggingface_hub import login
# from dotenv import load_dotenv
from transformers import AutoTokenizer
from LTSM.lstm import LSTMLanguageModel
from Transformer.transformer import Transformer
from Summary.summarization import summarize_lstm, summarize_decoder_only, summarize_encoder_decoder, summarize
from Translation.translator import translate_lstm, translate_decoder_only, translate_encoder_decoder, translate
from Tokenize.word_prediction import (
    get_texts,
    prepare_sequences,
    prepare_summarization_sequences,
    init_loss_fn,
    train_model,
    plot_loss,
    split_train_test,
    get_model_class,
)
from Config.config import cfg
from Tokenize.word_tokenizer import WordTokenizer
from data.xl_sum_dataset.xl_sum import preprocess_dataset_xl_sum,save_preprocessed_data_xl_sum,load_preprocessed_data_xl_sum, prepare_dataloader_xl_sum,count_json_lines_xl_sum,load_or_preprocess_xl_sum_data
from data.wmt14_dataset.wmt14 import preprocess_wmt14, prepare_dataloader
from Evaluation.bleu_score import calculate_bleu

# load_dotenv()
sys.path.append('src')

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
    test_data = preprocess_wmt14("test[:5%]", tokenizer)  # Use 5% of the test split for quick testing
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
    example_texts = [tokenizer.decode(src_list, skip_special_tokens=True) for src_list, _ in example_inputs]
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
        
    if cfg.save_model:
        print("Saving model...")
    # save model
    try : 
        torch.save(model.state_dict(), "models/model_translation.pth")
        print("The model has been saved successfully!")
    except Exception as e :
        print(f"failed with the error message: {e}") 
    # save tokenizer
    try :
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
    print('Summarization mode')
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
    train_data = load_or_preprocess_xl_sum_data("train[:1%]", tokenizer, dataset_dir, max_length=cfg.max_seq_length)
    test_data = load_or_preprocess_xl_sum_data("test[:1%]", tokenizer, dataset_dir, max_length=cfg.max_seq_length)
    
    # Prepare dataloaders
    print("Preparing dataloaders...")
    train_loader = prepare_dataloader_xl_sum(train_data, batch_size=cfg.batch_size, max_length=cfg.max_seq_length)
    test_loader = prepare_dataloader_xl_sum(test_data, batch_size=cfg.batch_size, max_length=cfg.max_seq_length)
    
    # Initialize model
    print("Initializing encoder-decoder Transformer model...")
    vocab_size = len(tokenizer)
    
    # Encoder-Decoder Transformer Model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout
    ).to(cfg.device)
    
    # Initialize loss function and optimizer
    criterion = init_loss_fn(cfg.loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Train model
    print("Training Encoder-Decoder Transformer model...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, tokenizer, optimizer, criterion, cfg
    )
    
    # Plot loss curves
    print("Plotting loss curves...")
    plot_loss(train_losses, test_losses, save_path="plots/transformer_summarization_loss.png")
    
    # Save model if configured
    if cfg.save_model:
        print("Saving model...")
        try:
            torch.save(model.state_dict(), "models/transformer_summarization.pth")
            print("Model saved successfully!")
        except Exception as e:
            print(f"Failed to save model with error: {e}")
        
        # Save tokenizer
        try:
            tokenizer.save_pretrained("models/summarization_tokenizer")
            print("Tokenizer saved successfully!")
        except Exception as e:
            print(f"Failed to save tokenizer with error: {e}")
        
        # Save config
        with open("models/config_summarization.json", "w") as f:
            json.dump(cfg.__dict__, f)
        print("Config saved to models/config_summarization.json")
    
    # Test summarization on example texts
    print("\nTesting summarization on example texts...")
    
    # Get a few examples from the test set
    example_inputs = []
    for batch in test_loader:
        src, tgt = batch
        # Convert tensors to lists directly using PyTorch
        src_list = src.squeeze(0).cpu().tolist()
        tgt_list = tgt.squeeze(0).cpu().tolist()
        example_inputs.append((src_list, tgt_list))
        if len(example_inputs) >= 3:  # Limit to 3 examples for testing
            break
    
    # Decode the example inputs for readability
    example_texts = []
    example_summaries = []
    for src, tgt in example_inputs:
        # Filter out padding tokens (0)
        src_tokens = [t for t in src if t != 0]
        tgt_tokens = [t for t in tgt if t != 0]
        
        # Convert token IDs to strings one at a time
        src_text = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(t) for t in src_tokens])
        tgt_text = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(t) for t in tgt_tokens])
        
        example_texts.append(src_text)
        example_summaries.append(tgt_text)
    
    print("\nExample Input Texts and Ground Truth Summaries:")
    for i, (text, summary) in enumerate(zip(example_texts, example_summaries)):
        print(f"\nExample {i+1}:")
        print(f"Input Text: {text}")
        print(f"Ground Truth Summary: {summary}")
    
    # Test summarization with the model
    print("\nTesting summarization with Encoder-Decoder Transformer:")
    for i, input_text in enumerate(example_texts):
        print(f"\nExample {i+1}:")
        print(f"Input Text: {input_text}")
        
        # Summarize using Encoder-Decoder Transformer
        _, generated_summary = summarize_encoder_decoder(model, input_text, tokenizer, cfg.device, max_length=50)
        print(f"Generated Summary: {generated_summary}")
        
        # Calculate BLEU score
        bleu_score = calculate_bleu([generated_summary], [example_summaries[i]])
        print(f"BLEU Score: {bleu_score:.4f}")
    
    print("-" * 50)



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
        try : 
            torch.save(model.state_dict(), "models/model_next_word_generator.pth")
            print("The model has been saved successfully!")
        except Exception as e :
            print(f"failed with the error message: {e}") 
        # save tokenizer
        try :
            torch.save(tokenizer, "models/tokenizer_next_word_generator.pth")
            print("The tokenizer has been saved successfully!")
        except Exception as e:
            print(f"failed with the error message: {e}")
        
        # save config
        with open("models/config_next_word_generator.json", "w") as f:
            json.dump(cfg, f)
        print("Model, tokenizer, and config saved to models")    


if __name__ == "__main__":
    
    if cfg.mode == "next-word-generation":
        next_word_generator()
    elif cfg.mode == "summarization":
        Summarization()
    elif cfg.mode == "translation":
        Translator()
    else:
        print(f"Unknown mode: {cfg.mode}. Available modes: next-word-generation, summarization, translation")
