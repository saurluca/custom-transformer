import torch
import torch.nn as nn
import torch.nn.functional as F
from Tokenize.word_tokenizer import WordTokenizer


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer(s)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]

        # Embed the input
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Pass through LSTM
        output, hidden = self.lstm(
            embedded, hidden
        )  # output: [batch_size, seq_len, hidden_dim]

        # Apply dropout
        output = self.dropout(output)

        # Pass through linear layer
        logits = self.fc(output)  # [batch_size, seq_len, vocab_size]

        # Return log probabilities
        return F.log_softmax(logits, dim=-1), hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden states
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
                device
            ),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
                device
            ),
        )
