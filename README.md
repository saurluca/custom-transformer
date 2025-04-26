# Custom Transformer: A Versatile Neural Architecture Implementation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)

## 📋 Overview

Custom Transformer is a comprehensive implementation of transformer-based neural architectures for various natural language processing tasks. This project provides a flexible framework for experimenting with different transformer variants, including encoder-decoder, decoder-only, and LSTM models, for tasks such as text summarization, translation, and next-word prediction.
You can find the paper [here](https://drive.google.com/file/d/1UbHl6D5zd0tBXhzIzMzIyhE1e9eAh65m/view?usp=share_link).

## ✨ Features

- **Multiple Model Architectures**:
  - Encoder-Decoder Transformer
  - Decoder-Only Transformer
  - LSTM-based Language Model

- **Versatile NLP Tasks**:
  - Text Summarization
  - Machine Translation
  - Next-Word Prediction

- **Comprehensive Evaluation**:
  - ROUGE metrics for summarization
  - BLEU score for translation
  - Perplexity for language modeling

- **Flexible Configuration**:
  - Customizable model parameters
  - Multiple tokenization options
  - Configurable training settings

## 🏗️ Architecture

The project is organized into several key components:

### Core Components

- **Transformer Module**: Implements the core transformer architecture with attention mechanisms
- **LSTM Module**: Provides a traditional recurrent neural network implementation
- **Tokenization**: Supports both custom word-level tokenization and Hugging Face tokenizers
- **Data Processing**: Handles various datasets including XL-sum for summarization and WMT14 for translation

### Model Variants

1. **Encoder-Decoder Transformer**:
   - Bidirectional encoder for understanding input context
   - Autoregressive decoder for generating output sequences
   - Cross-attention mechanism for connecting encoder and decoder

2. **Decoder-Only Transformer**:
   - Causal attention for autoregressive generation
   - Efficient for tasks where input and output share the same vocabulary

3. **LSTM Language Model**:
   - Traditional recurrent architecture
   - Baseline for comparing with transformer-based models

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Transformers library
- NLTK
- Matplotlib
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/custom-transformer.git
cd custom-transformer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('webtext'); nltk.download('gutenberg')"
```

### Configuration

The project uses a configuration file (`src/Config/config.py`) to manage settings:

```python
# Example configuration
cfg.model_type = "transformer"  # Options: "transformer", "decoder", "lstm"
cfg.d_model = 256              # Model dimension
cfg.num_heads = 8              # Number of attention heads
cfg.num_layers = 6             # Number of transformer layers
cfg.dropout = 0.1              # Dropout rate
cfg.mode = "summarization"     # Task mode: "summarization", "translation", "next-word-generation"
```

### Usage

#### Summarization

```python
from main import Summarization

# Run summarization with default settings
Summarization()
```

#### Translation

```python
from main import Translator

# Run translation with default settings
Translator()
```

#### Next-Word Generation

```python
from main import next_word_generator

# Run next-word generation with default settings
next_word_generator()
```

## 📊 Results

### Summarization Performance

| Model Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------|---------|---------|---------|
| LSTM | 0.15-0.20 | 0.05-0.10 | 0.12-0.17 |
| Decoder-Only | 0.18-0.23 | 0.07-0.12 | 0.15-0.20 |
| Encoder-Decoder | 0.20-0.25 | 0.08-0.13 | 0.17-0.22 |
| State-of-the-Art (BART) | 0.40-0.45 | 0.20-0.25 | 0.35-0.40 |

### Translation Performance

| Model Type | BLEU Score |
|------------|------------|
| LSTM | 15.2 |
| Decoder-Only | 18.7 |
| Encoder-Decoder | 22.3 |
| State-of-the-Art (T5) | 35.8 |

## 🔍 Analysis

### Model Comparison

- **LSTM Models**: Perform adequately for simple tasks but struggle with long-range dependencies and complex relationships.
- **Decoder-Only Transformers**: Excel at autoregressive tasks but may miss bidirectional context.
- **Encoder-Decoder Transformers**: Provide the best balance for sequence-to-sequence tasks like summarization and translation.

### Performance Insights

- **ROUGE Metrics**: Our models achieve 50-60% of the ROUGE scores compared to state-of-the-art models, indicating room for improvement in summary quality.
- **BLEU Scores**: Translation performance is approximately 60% of state-of-the-art models, suggesting that our custom implementation captures many but not all aspects of effective translation.
- **Training Efficiency**: LSTM models train faster but require more epochs to converge, while transformer models benefit from parallel processing but require more memory.

## 🛠️ Customization

### Adding New Models

To add a new model architecture:

1. Create a new class in the appropriate module (e.g., `src/Transformer/`)
2. Implement the required methods (`forward`, etc.)
3. Update the model factory in `src/Tokenize/word_prediction.py`
4. Add configuration options in `src/Config/config.py`

### Adding New Tasks

To add a new NLP task:

1. Create a new module in the `src/` directory
2. Implement task-specific functions (e.g., `summarize`, `translate`)
3. Add a new mode in `main.py`
4. Update the configuration to support the new task

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The transformer architecture was introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- This implementation is inspired by the [Harvard NLP's The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).
- Special thanks to the Hugging Face team for their excellent libraries and pre-trained models.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact one of the contributors:

Seyedalireza Yaghoubi:
[syaghoubi@uni-osnabreuck.de](mailto:Syaghoubi@uni-osnabrueck.de)

Luca Saur:
[mail@lucasaur.com](mailto:mail@lucasaur.com)

Jeelka Hessenius:
[jhessenius@uni-osnabrueck.de](mailto:jhessenius@uni-osnabrueck.de)

Florian Weigandt:
[fweigandt@uni-osnabrueck.de](mailto:fweigandt@uni-osnabrueck.de)





