import os
import torch


# text_file = "./data.txt"

# data = ""
# with open(text_file, "r") as file:
#     data = file.read()

# print(data)

class Tokenizer():

    def __init__(self, data):

       
        self.char2int = {}
        self.int2char = {}

        vocab = sorted(set(data))
        self.vocab_size = len(vocab)

        for idx, c in enumerate(vocab):
            self.char2int[c] = idx
            self.int2char[idx] = c 

    
    def encode(self,txt):
        encoded =[]
        for c in txt:
            encoded.append(self.char2int[c])
        return encoded


    def decode(self, encodd_list):
        output = "".join([self.int2char[i] for i in encodd_list])
        return output




# print(f"lenght of the dataset = {len(data)}")


# tk = Tokenizer(data)

# tokenized_data = torch.tensor(tk.encode(data), dtype=torch.long)

# print(tokenized_data[:100])

# context_length = 16
# batch_size = 4


def get_batch(tokenized_data, context_length, batch_size):

    dataset_length = len(tokenized_data)

    batch_start_index_list = torch.randint(0,dataset_length-1-context_length, (batch_size,))
    # print(batch_start_index_list)

    xx = torch.stack([ tokenized_data[i:i+context_length]  for i in batch_start_index_list])
    yy = torch.stack([ tokenized_data[i+context_length]  for i in batch_start_index_list])


    return xx,yy


# x, y = get_batch(tokenized_data, context_length, batch_size)
# print(x.shape , y.shape)
# print(x)
# print(y)