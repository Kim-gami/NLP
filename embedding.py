import torch
import torch.nn as nn

torch.manual_seed(0)

num_embeddings = 10 #max_sequence_length
embeddings_dim = 3 #batch_size

embedding = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = embeddings_dim)

input_tokens = torch.tensor([1, 5])
output_embeddings = embedding(input_tokens)
print(output_embeddings)
