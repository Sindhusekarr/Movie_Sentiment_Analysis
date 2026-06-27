import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Clean reviews
df["clean_review"] = df["review"].apply(clean_text)

# Create tokenizer
tokenizer = Tokenizer(num_words=5000)

# Learn vocabulary
tokenizer.fit_on_texts(df["clean_review"])

# Convert text into numbers
sequences = tokenizer.texts_to_sequences(df["clean_review"])

# Make all reviews the same length
padded_sequences = pad_sequences(sequences, maxlen=200)

print("Original Review:\n")
print(df["clean_review"][0])

print("\n--------------------------------")

print("\nTokenized Review:\n")
print(sequences[0][:30])

print("\n--------------------------------")

print("\nPadded Shape:")
print(padded_sequences.shape)