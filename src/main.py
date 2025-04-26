import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import matplotlib.pyplot as plt
import evaluate
from data.xl_sum_dataset.xl_sum import prepare_dataloader_xl_sum
from Config.config import cfg


def Summarization():
    print("\nStarting summarization task...")

    # Load tokenizer and set proper configuration
    print("Loading tokenizer: facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024  # Set maximum length for the tokenizer

    # Load and preprocess dataset
    print("Loading or preprocessing XL-sum dataset (small subset for testing)...")
    train_loader, test_loader = prepare_dataloader_xl_sum(tokenizer)

    # Initialize model
    print("Initializing encoder-decoder Transformer model...")
    model = AutoModelForSeq2SeqGeneration.from_pretrained("facebook/bart-large-cnn")
    model = model.to(cfg.device)

    # Train model
    print("Training Encoder-Decoder Transformer model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")
        model.train()
        total_loss = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                src, tgt = batch
                src = (
                    torch.tensor(src).to(cfg.device)
                    if isinstance(src, list)
                    else src.to(cfg.device)
                )
                tgt = (
                    torch.tensor(tgt).to(cfg.device)
                    if isinstance(tgt, list)
                    else tgt.to(cfg.device)
                )

                optimizer.zero_grad()
                outputs = model(input_ids=src, labels=tgt)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    src, tgt = batch
                    src = (
                        torch.tensor(src).to(cfg.device)
                        if isinstance(src, list)
                        else src.to(cfg.device)
                    )
                    tgt = (
                        torch.tensor(tgt).to(cfg.device)
                        if isinstance(tgt, list)
                        else tgt.to(cfg.device)
                    )

                    outputs = model(input_ids=src, labels=tgt)
                    loss = outputs.loss
                    total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}\n")

        # Generate a sample summary
        if epoch == 0:
            sample_text = "Peru prepares for El Nino"
            inputs = tokenizer(
                sample_text,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding=True,
            ).to(cfg.device)

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=128,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            print("\nGenerating samples:")
            print("Input Text:", sample_text)
            print(
                "Generated Summary:",
                tokenizer.decode(summary_ids[0], skip_special_tokens=True),
            )
            print("--------------------------------------------------")

    # Plot loss curves
    print("Plotting loss curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cfg.num_epochs + 1), [avg_train_loss], label="Training Loss")
    plt.plot(range(1, cfg.num_epochs + 1), [avg_test_loss], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.savefig("plots/transformer_summarization_loss.png")
    print("Loss plot saved to plots/transformer_summarization_loss.png")

    # Save model and tokenizer
    print("Saving model...")
    model.save_pretrained("models/transformer_summarization")
    tokenizer.save_pretrained("models/transformer_summarization")

    # Save config
    import json

    config = {
        "model_type": "encoder-decoder",
        "base_model": "facebook/bart-large-cnn",
        "max_length": tokenizer.model_max_length,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
    }
    with open("models/config_summarization.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Config saved to models/config_summarization.json")

    # Test the model on example texts
    print("\nTesting summarization on example texts...")
    print("\nExample Input Texts and Ground Truth Summaries:\n")

    example_texts = []
    example_summaries = []
    ground_truth_summaries = []

    # Get a few examples from the test dataset
    for i, batch in enumerate(test_loader):
        if i >= 4:  # Only process first 4 examples
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            src, tgt = batch
            src = torch.tensor(src) if isinstance(src, list) else src
            tgt = torch.tensor(tgt) if isinstance(tgt, list) else tgt

            # Convert source tokens back to text
            src_text = tokenizer.decode(src[0].tolist(), skip_special_tokens=True)
            tgt_text = tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)

            print(f"Example {i + 1}:")
            print(f"Input Text: {src_text}")
            print(f"Ground Truth Summary: {tgt_text}\n")

            example_texts.append(src_text)
            ground_truth_summaries.append(tgt_text)

    print("Testing summarization with Encoder-Decoder Transformer:\n")
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(example_texts):
            print(f"Example {i + 1}:")
            print(f"Input Text: {text}")

            # Tokenize input text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding=True,
            ).to(cfg.device)

            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=128,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"Generated Summary: {summary}\n")
            example_summaries.append(summary)

    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=example_summaries, references=ground_truth_summaries
    )

    print("\nROUGE Scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    # Plot ROUGE scores
    plt.figure(figsize=(10, 6))
    plt.bar(scores.keys(), scores.values())
    plt.title("ROUGE Scores")
    plt.ylabel("Score")
    plt.savefig("plots/rouge_scores.png")
    print("ROUGE scores plot saved to plots/rouge_scores.png")
    print("--------------------------------------------------")
