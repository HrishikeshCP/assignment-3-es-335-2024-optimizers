import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
from pprint import pprint
import requests
import time
from sklearn.manifold import TSNE
import os
import numpy as np

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

def visualize_embeddings_with_tsne(emb, stoi, itos, title='t-SNE Visualization of Embeddings', figsize=(10, 8), legend_loc='upper left'):
    """
    Visualizes embeddings with t-SNE.
    
    Args:
    - emb (torch.nn.Embedding): The embedding layer.
    - stoi (dict): Dictionary mapping characters to indices.
    - itos (dict): Dictionary mapping indices to characters.
    - title (str): Title of the plot (default: 't-SNE Visualization of Embeddings').
    - figsize (tuple): Figure size (default: (10, 8)).
    - legend_loc (str): Location of the legend (default: 'upper left').
    
    Returns:
    - None
    """
    # Get the embeddings from the embedding layer
    embeddings = emb.weight.data.numpy()

    # Initialize t-SNE with desired parameters
    tsne = TSNE(n_components=2, random_state=42)

    # Fit t-SNE to the embeddings
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Define a color map for distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(itos)))

    # Visualize the t-SNE embeddings
    plt.figure(figsize=figsize)
    for i in range(len(embeddings)):
        plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], label=itos[i], color=colors[i], marker='.')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Characters', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()
    return plt