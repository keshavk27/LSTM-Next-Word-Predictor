import streamlit as st
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e1e2f, #2c3e50);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 10px;
    animation: fadeIn 1.2s ease-in;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 30px;
}

.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 12px;
}

.stButton button {
    border-radius: 12px;
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    font-size: 16px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #43cea2, #185a9d);
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background: #262730;
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
    animation: fadeIn 0.8s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_resources():
    model = load_model('lstm_next_word.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

# Reverse word index (FAST lookup)
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    return reverse_word_index.get(predicted_word_index, "❓")

# ---------------- UI ----------------
st.markdown('<div class="main-title">✨ Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Powered by LSTM + Early Stopping</div>', unsafe_allow_html=True)

input_text = st.text_input(
    "Enter the sentence:",
    "To be or not to"
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_btn = st.button("🚀 Predict Next Word")

if predict_btn:
    with st.spinner("Thinking... 🤖"):
        time.sleep(1)  # animation feel

        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    st.markdown(
        f'<div class="result-box">Predicted Word: <b>{next_word}</b></div>',
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown("""
<br><hr>
<center style='color:gray;'>Built  using Streamlit & TensorFlow</center>
""", unsafe_allow_html=True)