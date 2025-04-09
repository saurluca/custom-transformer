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
        return self.embedding(x)* math.sqrt(self.d_model)


    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # Create a matrix of zeros of dimensions max_seq_length by d_model
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Perform the sine and cosine calculations
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Ensure pe isn't a learnable parameter during training
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add the positional embeddings to the token embeddings
        return x + self.pe[:, :x.size(1)]


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
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
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


def main():
    print("Starting transformation ...")

    token_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    
    print("Embeding tokens")
    # Instantiate InputEmbeddings and apply it to token_ids
    embedding_layer = InputEmbeddings(vocab_size=10000, d_model=512)
    token_embeddings = embedding_layer(token_ids)
    print(token_embeddings.shape)

    print("Adding postional encoding")
    pos_encoding_layer = PositionalEncoding(d_model=512, max_seq_length=4)
    output = pos_encoding_layer(token_embeddings)
    print(output.shape)
    print(output[0][0][:10])
    
    print("Done!")



if __name__ == "__main__":
    main()