import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df["clean_review"] = df["review"].apply(clean_text)

# Tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["clean_review"])

X = tokenizer.texts_to_sequences(df["clean_review"])
X = pad_sequences(X, maxlen=200)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["sentiment"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = Sequential()

model.add(Embedding(5000, 128))

model.add(LSTM(64))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)
import pickle

# Save the trained model
model.save("models/sentiment_model.keras")

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Model saved successfully!")