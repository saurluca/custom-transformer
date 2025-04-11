from types import SimpleNamespace
import torch

cfg = SimpleNamespace(**{})

# run config
cfg.save_model = False
cfg.generate_samples = False

# data
cfg.text_names = [
    "gutenberg-austen-emma.txt",
    # "webtext-grail.txt",
]  # "webtext-overheard.txt", "gutenberg-austen-emma.txt"

cfg.num_samples = 10000
cfg.max_chars_per_text = 200000  # Limit each text to num of characters
cfg.min_vocab_freq = 2
cfg.max_vocab_size = 10000
cfg.seq_length = 32
cfg.train_size = 0.9
cfg.use_pretrained = False
cfg.pretrained_model = "bert-base-uncased"

# training
cfg.batch_size = 128
cfg.num_epochs = 3
cfg.num_workers = 2
cfg.learning_rate = 0.0001
cfg.weight_decay = 0.0001
cfg.loss_fn = "CrossEntropyLoss"  # "CrossEntropyLoss", "NLL"

# model
cfg.d_model = 256
cfg.num_layers = 8
cfg.num_heads = 8
cfg.d_ff = 1024  # recommended: 4x d_model
cfg.dropout = 0.1
cfg.max_seq_length = 32

# text generation
cfg.output_length = 24  # max length of generated text
cfg.seq_length_gen = 32  # sequence length for generation
cfg.temperature = 1.5
cfg.top_k = 15
cfg.top_p = 0.5
cfg.sampling_strategy = "multinomial"  # "multinomial", "greedy", "top-k", "top-p"
cfg.example_prompts = ["The man who", "Pizza guy:", "I love "]
cfg.show_top_k = False

cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
