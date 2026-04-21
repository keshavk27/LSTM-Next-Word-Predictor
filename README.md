# ✨ Next Word Prediction using LSTM

A modern **Next Word Prediction Web App** built using **LSTM (Long Short-Term Memory)** neural networks and deployed with **Streamlit**.
The model predicts the most probable next word for a given input sequence, similar to autocomplete systems.

---

## 🚀 Demo

Enter a phrase like:

> *"This is all about "*

👉 The model predicts the most likely next word based on learned patterns.

---

## 🧠 Features

* 🔮 Predicts the **next word** using a trained LSTM model
* ⚡ Fast inference with optimized tokenizer lookup
* 🎨 **Modern UI** with animations and custom styling
* ⏳ Loading spinner for better user experience
* 📦 Easy to deploy using Streamlit

---

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Backend/Model**: TensorFlow / Keras
* **Language**: Python
* **Libraries**:

  * NumPy
  * Pickle
  * TensorFlow
  * Streamlit

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── lstm_next_word.h5       # Trained LSTM model
├── tokenizer.pickle       # Tokenizer used during training
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/keshavk27/LSTM-Next-Word-Predictor
```

### 2️⃣ Create virtual environment (recommended)

```bash
conda create -p nlp_env python=3.10
conda activate nlp_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:{PORT}
```

---

## 🧪 How It Works

1. Input text is converted into a sequence using a tokenizer
2. Sequence is padded to match model input size
3. LSTM model predicts probability distribution over vocabulary
4. Word with highest probability is selected as output

---

## 📸 UI Preview

* Clean gradient UI
* Animated buttons
* Result card display
* Loading spinner

---

## 🔥 Future Improvements

* Show **top-k predictions** instead of one
* Add **confidence scores**
* Implement **real-time autocomplete typing**
* Train on larger datasets (e.g., books, Wikipedia)
* Deploy on **Streamlit Cloud / AWS / GCP**

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---



## 👨‍💻 Author

**Keshav Kumar**

* Passionate about AI, ML, and problem solving
* Competitive Programming Enthusiast

---

## 💡 Acknowledgements

* TensorFlow & Keras Documentation
* Streamlit Community

---

⭐ If you found this project useful, consider giving it a star!
