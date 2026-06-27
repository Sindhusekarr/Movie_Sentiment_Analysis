import pandas as pd

# Read the dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nShape of Dataset:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Count positive and negative reviews
print("\nSentiment Count:")
print(df["sentiment"].value_counts())