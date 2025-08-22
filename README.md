# NLP Foundations Sprint: IMDB Sentiment Analysis Engine v1

A foundational NLP sentiment analysis pipeline built in a 7-day sprint. This project demonstrates core AI engineering skills from data preprocessing to model evaluation as part of my self-driven "PRIME Protocol" to master the fundamentals.

---

## Objective

The goal of this project was to build a complete, end-to-end machine learning pipeline capable of classifying movie reviews from the IMDB dataset as either 'positive' or 'negative'. This sprint served as a practical application of foundational NLP and machine learning concepts.

---

## Methodology

The project follows a standard, four-step machine learning workflow:

1.  **Data Loading & Exploration:** The 50,000-review IMDB dataset was loaded using the Pandas library. Initial analysis confirmed a perfectly balanced dataset, ideal for training an unbiased classifier.

2.  **Preprocessing & Feature Extraction:** The raw text data was split into an 80% training set and a 20% testing set. I used Scikit-learn's `TfidfVectorizer` to convert the text into a numerical format, which intelligently weighs the importance of each word.

3.  **Model Training:** A `LogisticRegression` model from Scikit-learn was trained on the vectorized training data. This model was chosen for its efficiency and interpretability as a strong baseline classifier.

4.  **Evaluation:** The trained model's performance was evaluated on the 20% of unseen test data.

---

## Results

The model was evaluated based on its accuracy in predicting the sentiment of unseen reviews.

* **Final Model Accuracy:** `89.99%`

This result demonstrates a strong foundational ability to build a functional and effective predictive model.

---

## 🚀 Technologies Used

* **Language:** Python
* **Libraries:** Pandas, Scikit-learn
* **Tools:** Git, GitHub, VS Code

---



