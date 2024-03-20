# Streamlit UI
import streamlit as st

st.title("Next Character Predictor")

input_text = st.text_input("Enter your input text:")
k = st.slider("Number of characters to predict:", min_value=1, max_value=20, value=5)

if st.button("Predict"):
    if input_text:
        predicted_text = generate_text(input_text, model, itos, stoi, block_size, k)
        st.write("Predicted Text:", predicted_text)
    else:
        st.warning("Please enter some input text.")