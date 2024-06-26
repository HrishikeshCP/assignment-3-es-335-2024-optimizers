# -*- coding: utf-8 -*-
"""q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KhTAVkumKG7DMPtucjmSSjHvW66r_lLc
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from pprint import pprint
import requests
import time
import os

##################################################################
block_size = 10# context length: how many characters do we take to predict the next one?
emb_dim = 15
#################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

#Download Shakespeare.txt file

url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt"
file_path = "shakespeare.txt"
if os.path.exists(file_path):
    print("File already exists.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

with open("shakespeare.txt", "r") as file:
    content = file.read()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(content)))
stoi = {s:i for i,s in enumerate(chars)}
# stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
pprint(itos)

#  block_size = 10# context length: how many characters do we take to predict the next one
X, Y = [], []


#print(w)
context = [1] * block_size
for ch in content:
  ix = stoi[ch]
  X.append(context)
  Y.append(ix)
  print(''.join(itos[i] for i in context), '--->', itos[ix])
  context = context[1:] + [ix] # crop and append

# Move data to GPU

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

X.shape, X.dtype, Y.shape, Y.dtype

# emb_dim = 15
emb = torch.nn.Embedding(len(stoi), emb_dim)
emb.weight
emb.weight.shape

# Function to visualize the embedding in 2d space

# def plot_emb(emb, itos, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     for i in range(len(itos)):
#         x, y = emb.weight[i].detach().cpu().numpy()
#         ax.scatter(x, y, color='k')
#         ax.text(x + 0.05, y + 0.05, itos[i])
#     return ax

# plot_emb(emb, itos)

class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, hidden_size)
    self.lin3 = nn.Linear(hidden_size, hidden_size)
    self.lin4 = nn.Linear(hidden_size, vocab_size)


  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = torch.sin(self.lin2(x))
    x = torch.sin(self.lin3(x))
    x = self.lin4(x)
    return x

# Generate Text from untrained model


model = NextChar(block_size, len(stoi), emb_dim, 512).to(device)
# model = torch.compile(model)

g = torch.Generator()
g.manual_seed(4000002)
def generate_text(model, itos, stoi, block_size,iptxt="",max_len=10):
    context=[]
    if len(iptxt)<block_size:
      context=[1]*(block_size-len(iptxt))
      for char in iptxt:
        idx=stoi[char]
        context.append(idx)
    else:
      for i in range(len(iptxt)-block_size,len(iptxt)):
          idx=stoi[iptxt[i]]
          context.append(idx)
    txt = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        txt += ch
        context = context[1:] + [ix]
    return txt

print(generate_text(model, itos, stoi, block_size))

# Check if the model weights file exists
weights_file = f"model_weights_b{block_size}_em{emb_dim}.pth"
if os.path.exists(weights_file):
    # Load the saved model weights
    model.load_state_dict(torch.load(weights_file))
    print("Model weights loaded successfully.")
else:
    #code for training
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.005)
    import time
    # Mini-batch training
    batch_size = 4096
    print_every = 100
    elapsed_time = []
    for epoch in range(10000):
        start_time = time.time()
        for i in range(0, X.shape[0], batch_size):
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        if epoch % print_every == 0:
            print(epoch, loss.item())

    # Save the model weights
    torch.save(model.state_dict(), f"model_weights_b{block_size}_em{emb_dim}.pth")
    print("Model weights saved successfully.")

# Load the saved model weights
model.load_state_dict(torch.load(f"model_weights_b{block_size}_em{emb_dim}.pth"))

# Visualize the embedding

# plot_emb(model.emb, itos)

#generate text
inp='BRUTUS'
print(generate_text(model, itos, stoi,block_size,inp,1000+len(inp)))