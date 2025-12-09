import os
import torch
import torch.nn.functional as F
from data import Tokenizer, get_batch
from model import SimpleMLP


seq_len = 16
batch_size = 4
embedding_dim = 32
hidden_size = 1024


text_file = "./data.txt"

data = ""
with open(text_file, "r") as file:
    data = file.read()

tk = Tokenizer(data)

tokenized_data = torch.tensor(tk.encode(data), dtype=torch.long)


vocab_size = tk.vocab_size

model = SimpleMLP(vocab_size, seq_len, embedding_dim, hidden_size)

 



device = "cpu"

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

step_size = 15000

for step in range(step_size):

    xx, yy = get_batch(tokenized_data, seq_len, batch_size)
    xx = xx.to(device)
    yy = yy.to(device)

    output_logits = model(xx)

    loss = F.cross_entropy(output_logits, yy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%100==0:
        print(f"step={step}, loss={loss.item():.4f}")



def generate(model, seq_len, start_token, max_output_token=50):
    
    model.eval()

    output_tokens = start_token
    print(output_tokens.shape)
    
    with torch.no_grad():

        for i in range(max_output_token):

            
            context = output_tokens[:,-seq_len:]
            # print(f"context.shape={context.shape}")
            output_logits = model(context)
            # print(f"output_logits.shape={output_logits.shape}")
            # last_output_logit = output_logits[-1]

            predicted_token_id = torch.argmax(output_logits, dim=-1)

            # print(f"predicted_token_id={predicted_token_id}")
            # print(f"output_tokens.shape={output_tokens}")
            # print(f"predicted_token_id.shape={predicted_token_id}")

            output_tokens = torch.cat([output_tokens, predicted_token_id.unsqueeze(0)], dim=-1)
            # print(f"output_tokens={output_tokens}")

        return output_tokens.squeeze(0)


start_string = "once"
start_token = torch.tensor(tk.encode(start_string), dtype=torch.long).to(device)

pad_len = seq_len-start_token.shape[0]
pad = start_token[0].repeat(pad_len)
new_start_token = torch.cat([start_token, pad])
new_start_token = new_start_token.unsqueeze(0)



output = generate(model,seq_len, new_start_token)
output = output.tolist()
print(f"output = {tk.decode(output)}")