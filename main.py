import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  
# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Print the first 5 rows to verify the data has been loaded
print("Dataset loaded successfully. First 5 rows:")
print(df.head())

# Print the number of positive vs. negative reviews
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Define features (X) and target (y)
X = df['review']
y = df['sentiment']

# Data Split: Split data into training and testing sets
# This holds back 20% of the data for a final test of our model's accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize: Turn the text into numbers
# Create the vectorizer tool
vectorizer = TfidfVectorizer()

# Learn the vocabulary from the training data and transform it into numbers
X_train_vec = vectorizer.fit_transform(X_train)

# Only transform the test data using the vocabulary learned from the training data
X_test_vec = vectorizer.transform(X_test)

print("\nData successfully split and vectorized.")
print("Shape of training data:", X_train_vec.shape)
print("Shape of testing data:", X_test_vec.shape)

model = LogisticRegression()

# Train the model on vectorized training data
print("\nTraining model...")
model.fit(X_train_vec, y_train)
print("Model training complete.")

# Evaluate the Model
# Use the trained model to make predictions on the test data
predictions = model.predict(X_test_vec)

# Compare the model's predictions to the actual correct answers
accuracy = accuracy_score(y_test, predictions)

print("\n--- Mission Report ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("--------------------")