import torch
import torch.nn as nn
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