import streamlit as st
import torch
from q1fns import generate_text,NextChar


# Streamlit UI
st.title("Next Character Predictor")

itos={0: '\n',
 1: ' ',
 2: '!',
 3: "'",
 4: ',',
 5: '-',
 6: '.',
 7: ':',
 8: ';',
 9: '?',
 10: 'A',
 11: 'B',
 12: 'C',
 13: 'D',
 14: 'E',
 15: 'F',
 16: 'G',
 17: 'H',
 18: 'I',
 19: 'J',
 20: 'K',
 21: 'L',
 22: 'M',
 23: 'N',
 24: 'O',
 25: 'P',
 26: 'Q',
 27: 'R',
 28: 'S',
 29: 'T',
 30: 'U',
 31: 'V',
 32: 'W',
 33: 'X',
 34: 'Y',
 35: 'Z',
 36: 'a',
 37: 'b',
 38: 'c',
 39: 'd',
 40: 'e',
 41: 'f',
 42: 'g',
 43: 'h',
 44: 'i',
 45: 'j',
 46: 'k',
 47: 'l',
 48: 'm',
 49: 'n',
 50: 'o',
 51: 'p',
 52: 'q',
 53: 'r',
 54: 's',
 55: 't',
 56: 'u',
 57: 'v',
 58: 'w',
 59: 'x',
 60: 'y',
 61: 'z'}
block_size=10
emb_dim=15
stoi = {i:s for s,i in itos.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NextChar(block_size, len(stoi), emb_dim, 100).to(device)
model.load_state_dict(torch.load("qn1/model_weights.pth",map_location=device))
block_size = st.selectbox('Context size',[5,10,15])
emb_dim = st.selectbox('Embedding size',[2,5,8,12,15])
input_text = st.text_input("Enter your input text:")
k = st.slider("Number of characters to predict:", min_value=1, max_value=20, value=5)
if st.button("Predict"):
    if input_text:
        predicted_text = generate_text( model, itos, stoi, block_size,input_text, k)
        st.write("Predicted Text:", input_text+predicted_text)
    else:
        st.warning("Please enter some input text.")