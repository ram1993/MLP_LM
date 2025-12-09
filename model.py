import torch
import torch.nn as nn 
import torch.nn.functional as F
from data import get_batch, Tokenizer
class SimpleMLP(nn.Module):

    def __init__(self, vocab_size, max_seq_len, embedding_dim, hidden_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.fc1 = nn.Linear(max_seq_len*embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    
    
    def forward(self, input_ids):
        
        batch_size, seq_len = input_ids.shape

        token_emb = self.token_embedding(input_ids) # BxS
        pos = torch.arange(seq_len, dtype=torch.long, device="cpu")
        pos_emb = self.pos_embedding(pos) # S,

        input_emb = token_emb+pos_emb
        # print(token_emb.shape, pos_emb.shape, input_emb.shape)

        input_emb = input_emb.reshape(batch_size,-1)

        out1 = F.relu(self.fc1(input_emb))
        logits = self.fc2(out1)
        # print(out1.shape)
        # print(logits.shape)

        return logits



# seq_len = 16
# batch_size = 4
# embedding_dim = 32
# hidden_size = 1024


# text_file = "./data.txt"

# data = ""
# with open(text_file, "r") as file:
#     data = file.read()

# tk = Tokenizer(data)

# tokenized_data = torch.tensor(tk.encode(data), dtype=torch.long)

# x, y = get_batch(tokenized_data, seq_len, batch_size)
# print(x)
# print(y)
# print(x.shape, y.shape)

# vocab_size = tk.vocab_size

# model = SimpleMLP(vocab_size, seq_len, embedding_dim, hidden_size)
# output = model(x)
# print(output.shape)
 