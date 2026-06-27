import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text

# Apply cleaning to all reviews
df["clean_review"] = df["review"].apply(clean_text)

# Display original and cleaned reviews
print("\nOriginal Review:\n")
print(df["review"][0])

print("\n-------------------------------------")

print("\nCleaned Review:\n")
print(df["clean_review"][0])