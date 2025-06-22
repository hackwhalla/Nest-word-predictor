import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

# ✅ Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    model = load_model('Nest_word_predictor1.keras')
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ✅ Reverse lookup for predictions
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

# ✅ Prediction function
def predict_next_words(text):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([token_text], maxlen=25, padding='pre')
    prediction = model.predict(padded, verbose=0)[0]
    top_3_idx = np.argsort(prediction)[-3:][::-1]
    top_words = [index_to_word[i] for i in top_3_idx if i in index_to_word]
    return top_words

# ✅ Streamlit UI
st.title("Next Word Prediction using LSTM based on all previous words")
st.markdown("👉 Type your sentence below and click **Predict** to get next 3 word predictions.")

# Input box
user_input = st.text_input("Enter your sentence:", value="", max_chars=200)

# Predict button
if st.button("🔮 Predict"):
    if user_input.strip():
        predictions = predict_next_words(user_input.strip())
        st.markdown("### ➤ Predicted Next Words:")
        st.success(", ".join(predictions))
    else:
        st.warning("⚠️ Please enter a sentence to predict.")
