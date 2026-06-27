import streamlit as st
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* Analyze Sentiment Button */

div.stButton > button {
    background-color: #16A34A !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: bold !important;
    height: 60px !important;
    border-radius: 12px !important;
    border: none !important;
}

div.stButton > button:hover {
    background-color: #15803D !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #202124;
    color: white;
}

/* Main content */
.main {
    background-color: #202124;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    color: white;
}

/* Headings */
h1 {
    color: #E50914;
    text-align: center;
    font-size: 3rem;
}

h2, h3, h4, h5, h6 {
    color: white;
}

/* Normal text */
p, label, div {
    color: white;
}

/* Text Area */
textarea {
    background-color: #2B2B2B !important;
    color: white !important;
    border: 2px solid #16A34A !important;
    border-radius: 10px;
}

textarea::placeholder {
    color: #FFFFFF !important;
    opacity: 0.6 !important;
}

/* Button */
div.stButton > button {
    background: #16A34A !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 55px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

div.stButton > button:first-child {
    font-size: 30px !important;
}
div.stButton > button:hover {
    background: #15803D !important;
    color: white !important;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-color:#202124;
}

/* Success Message */
div[data-testid="stAlert"] {
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Analyze Sentiment Button */
div.stButton > button {
    background-color: #16A34A !important;
    color: white !important;
    height: 65px !important;
    border-radius: 12px !important;
    border: none !important;
}

div.stButton > button p {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: white !important;
}

div.stButton > button:hover {
    background-color: #15803D !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 About Project")

st.sidebar.markdown("""
### 👩 Developer
**Sindhu**

### 🧠 Model
LSTM Neural Network

### 🎥 Dataset
IMDb 50K Movie Reviews

### 📈 Accuracy
**88%**

### 🛠 Technologies
- Python
- TensorFlow
- Streamlit
- Keras
""")

# -----------------------------
# Load Model
# -----------------------------
model = load_model("models/sentiment_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# -----------------------------
# Header
# -----------------------------
st.markdown("# 🎬 AI Movie Review Analyzer 🎥")

st.markdown("""
<h3 style="text-align:center; color:white;">
😍 Love It or 😴 Skip It?
</h3>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Input
# -----------------------------
st.markdown("""
<style>

/* Your existing CSS */

/* Placeholder text */
div[data-baseweb="textarea"] textarea::placeholder {
    font-size: 18px !important;
    color: #CFCFCF !important;
    opacity: 1 !important;
}

/* Text typed by user */
div[data-baseweb="textarea"] textarea {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown(
    "<p style='color:white; font-size:20px; font-weight:bold;'>📝 Enter your movie review</p>",
    unsafe_allow_html=True
)

review = st.text_area(
    "",
    height=200,
    placeholder="Example: This movie was absolutely amazing. I loved every scene..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("⚠ Please enter a movie review.")
    else:

        review = clean_text(review)

        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=200)

        prediction = model.predict(padded, verbose=0)

        confidence = float(prediction[0][0])

        st.divider()

        st.subheader("🎯 Prediction")

        if confidence >= 0.5:
            sentiment = encoder.inverse_transform([1])[0]
            confidence = confidence * 100

            st.success("😊 Positive Review")
           

        else:
            sentiment = encoder.inverse_transform([0])[0]
            confidence = (1 - confidence) * 100

            st.error("😞 Negative Review")
            
        st.write("### 📊 Confidence Score")

        st.progress(confidence / 100)

        st.write(f"**{confidence:.2f}%**")

        if confidence >= 90:
            st.write("⭐⭐⭐⭐⭐ Excellent Confidence")
        elif confidence >= 80:
            st.write("⭐⭐⭐⭐ Very Good Confidence")
        elif confidence >= 70:
            st.write("⭐⭐⭐ Good Confidence")
        else:
            st.write("⭐⭐ Fair Confidence")

st.divider()

st.info("💡 Tip: Try both positive and negative movie reviews to see how the AI predicts sentiment.")

st.divider()

st.caption("❤️ Developed by Sindhu using TensorFlow, Keras & Streamlit")