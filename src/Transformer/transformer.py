import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # Set the model dimensionality and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Instantiate the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Return the embeddings multiplied by the square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # Create a matrix of zeros of dimensions max_seq_length by d_model
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Perform the sine and cosine calculations
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Ensure pe isn't a learnable parameter during training
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # Add the positional embeddings to the token embeddings
        # Ensure the positional encoding is properly sized for the input sequence
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # If sequence length is greater than max_seq_length, extend the positional encoding
            self.pe = F.pad(self.pe, (0, 0, 0, seq_len - self.pe.size(1)))
        return x + self.pe[:, :seq_len]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        # Split the input embeddings and permute
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None):
        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def combine_heads(self, x, batch_size):
        # Combine heads back to (batch_size, seq_length, d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Build the forward pass
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention_weights = self.compute_attention(query, key, value, mask)
        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Define the layers and activation
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input through the layers and activation
        return self.fc2(self.relu(self.fc1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        # Perform the attention calculation
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Apply dropout and the first layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        # Pass through the feed-forward sublayer
        ff_output = self.ff_sublayer(x)
        # Apply dropout and the second layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayerWithCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Define cross-attention and a third layer normalization
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, tgt_mask, cross_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        # Complete the forward pass
        cross_attn_output = self.cross_attn(x, y, y, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length
    ):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # Define the list of decoder layers and linear layer
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # Define a linear layer to project hidden states to likelihoods
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt_mask):
        # Complete the forward pass
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class TransformerDecoderWithCrossAttention(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length
    ):
        super(TransformerDecoderWithCrossAttention, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # Define the list of decoder layers and linear layer
        self.layers = nn.ModuleList(
            [
                DecoderLayerWithCrossAttention(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        # Define a linear layer to project hidden states to likelihoods
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, y, tgt_mask, cross_mask):
        # Complete the forward pass
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, y, tgt_mask, cross_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length
    ):
        super().__init__()
        # Define the embedding, positional encoding, and encoder layers
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        # Perform the forward pass through the layers
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# Complete the classification head
class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        # Instantiate the layers
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Complete the forward method
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length
        )
        self.decoder = TransformerDecoderWithCrossAttention(
            vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length
        )

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, cross_mask)
        return decoder_output
