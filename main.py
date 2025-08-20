# Import the necessary library
import pandas as pd

# Load the dataset from the CSV file
df = pd.read_csv('IMDB Dataset.csv')

# Print the first 5 rows to verify the data has been loaded
print("Dataset loaded successfully. First 5 rows:")
print(df.head())

# Print the number of positive vs. negative reviews
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())