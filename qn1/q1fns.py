import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
from pprint import pprint
import requests
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate Text from untrained model
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

# model = NextChar(block_size, len(stoi), emb_dim, 100).to(device)
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

# print(generate_text(model, itos, stoi, block_size))