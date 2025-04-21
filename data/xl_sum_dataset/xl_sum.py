import torch
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def preprocess_dataset_xl_sum(split, tokenizer, max_length=100):
    """
    Preprocess the WMT14 dataset for training.

    Args:
        split (str): Dataset split to preprocess ("train", "validation", "test").
        tokenizer: Tokenizer instance for tokenizing text.
        max_length (int): Maximum sequence length.

    Returns:
        List of tokenized input-output pairs.
    """
    # Load the xl-sum dataset
    dataset =load_dataset("csebuetnlp/xlsum",name='english', split = split)

    tokenized_data = []
    for example in tqdm(dataset, desc=f"Preprocessing {split} split"):
        # Extract source and target sentences dynamically
        src_text = example["title"]
        tgt_text = example["summary"]

        # Tokenize 
        src_tokens = tokenizer.encode(src_text, max_length=max_length)
        tgt_tokens = tokenizer.encode(tgt_text, max_length=max_length)

        tokenized_data.append((src_tokens, tgt_tokens))

    return tokenized_data

def save_preprocessed_data_xl_sum(data, save_path):
    """
    Save preprocessed data to a JSON file.

    Args:
        data: List of tokenized input-output pairs.
        save_path (str): Path to save the JSON file.
    """
    directory = os.path.dirname(save_path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for src_tokens, tgt_tokens in data:
            json.dump({"src": src_tokens, "tgt": tgt_tokens}, f)
            f.write("\n")

def load_preprocessed_data_xl_sum(file_path, lines):
    """
    Load preprocessed data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List of tokenized input-output pairs.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data", total=lines):
            example = json.loads(line)
            data.append((example["src"], example["tgt"]))
    return data

def prepare_dataloader_xl_sum(data, batch_size=32, max_length=50):
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
        for seq in tqdm(targets_list, desc="Padding Inputs")
    )

    # Convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(inputs, targets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def count_json_lines_xl_sum(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_or_preprocess_xl_sum_data(
    split,
    tokenizer,
    dataset_dir,
    max_length=100,
):
    """
    Load preprocessed xl-sum data if it exists, otherwise preprocess and save it.

    Args:
        split (str): Dataset split to load or preprocess (e.g., "train", "validation", "test[:5%]").
        tokenizer: Tokenizer instance for tokenizing text.
        dataset_dir (str): Directory to save or load preprocessed data.
        source_lang (str): Source language code (e.g., "en").
        target_lang (str): Target language code (e.g., "de").
        max_length (int): Maximum sequence length.

    Returns:
        List of tokenized input-output pairs.
    """
    # Construct the file path for the preprocessed data
    file_path = os.path.join(dataset_dir, f"{split}_summarization.json")

    # Check if the preprocessed data exists
    if os.path.exists(file_path):
        print(f"Loading preprocessed XL-sum data from {file_path}...")
        num_lines = count_json_lines_xl_sum(file_path)
        return load_preprocessed_data_xl_sum(file_path, lines=num_lines)

    # If not, preprocess the data
    print(f"Preprocessing XL-sum dataset ({split}) for summarization...")
    data = preprocess_dataset_xl_sum(split, tokenizer, max_length)

    # Save the preprocessed data
    save_preprocessed_data_xl_sum(data, file_path)
    print(f"Preprocessed WMT14 data saved to {file_path}.")

    return data
