import torch
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import os 
from tqdm import tqdm 

def preprocess_wmt14(split, tokenizer, max_length=50):
    """
    Preprocess the WMT14 dataset for training.

    Args:
        split (str): Dataset split to preprocess ("train", "validation", "test").
        tokenizer: Tokenizer instance for tokenizing text.
        max_length (int): Maximum sequence length.

    Returns:
        List of tokenized input-output pairs.
    """
    # Load the WMT14 dataset
    dataset = load_dataset("wmt14", "de-en", split=split)

    tokenized_data = []
    for example in tqdm(dataset, desc=f"Preprocessing {split} split"):
        # Extract English and German sentences
        src_text = example["translation"]["en"]
        tgt_text = example["translation"]["de"]

        # Tokenize and truncate
        src_tokens = tokenizer.encode(src_text, truncation=True, max_length=max_length)
        tgt_tokens = tokenizer.encode(tgt_text, truncation=True, max_length=max_length)

        tokenized_data.append((src_tokens, tgt_tokens))

    return tokenized_data

def save_preprocessed_data(data, save_path):
    """
    Save preprocessed data to a JSON file.

    Args:
        data: List of tokenized input-output pairs.
        save_path (str): Path to save the JSON file.
    """
    directory = os.path.dirname(save_path)
    if not os.path.isdir(directory):
        os.makedirs(directory,exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for src_tokens, tgt_tokens in data:
            json.dump({"src": src_tokens, "tgt": tgt_tokens}, f)
            f.write("\n")

def load_preprocessed_data(file_path, lines):
    """
    Load preprocessed data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List of tokenized input-output pairs.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data" , total=lines):
            example = json.loads(line)
            data.append((example["src"], example["tgt"]))
    return data

def prepare_dataloader(data, batch_size=32, max_length=50):
    """
    Prepare a DataLoader for training.

    Args:
        data: List of tokenized input-output pairs.
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum sequence length.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    inputs_list, targets_list = zip(*data)

    inputs = []
    targets = []

    inputs.extend(
        seq + [0] * (max_length - len(seq))
        for seq in tqdm(inputs_list, desc="Padding Inputs")
    )
    targets.extend(
        seq + [0] * (max_length - len(seq))
        for seq in tqdm(inputs_list, desc="Padding Inputs")
    )

    # Convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return tqdm(data_loader, desc='Creating Batches for converting data to tensors') 

def count_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


# Example usage
tokenizer_name = "Helsinki-NLP/opus-mt-de-en"
dataset_file_path = "datasets/wmt14/train.json"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
if not os.path.exists(dataset_file_path):
    train_data = preprocess_wmt14("train", tokenizer)
    save_preprocessed_data(train_data, dataset_file_path)
num = count_json_lines(dataset_file_path)
train_data = load_preprocessed_data(dataset_file_path , lines = num)
train_loader = prepare_dataloader(train_data)
