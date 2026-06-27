import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("models/sentiment_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Take user input
review = input("Enter a movie review:\n\n")

# Clean
review = clean_text(review)

# Convert to sequence
sequence = tokenizer.texts_to_sequences([review])

# Padding
padded = pad_sequences(sequence, maxlen=200)

# Prediction
prediction = model.predict(padded, verbose=0)

# Probability
confidence = float(prediction[0][0])

if confidence >= 0.5:
    sentiment = encoder.inverse_transform([1])[0]
    confidence = confidence * 100
else:
    sentiment = encoder.inverse_transform([0])[0]
    confidence = (1 - confidence) * 100

print("\nPrediction :", sentiment)
print(f"Confidence : {confidence:.2f}%")