![GitHub Repo stars](https://img.shields.io/github/stars/Omkar10010/imdb-sentiment-analysis?style=social)
![GitHub forks](https://img.shields.io/github/forks/Omkar10010/imdb-sentiment-analysis?style=social)
![GitHub license](https://img.shields.io/github/license/Omkar10010/imdb-sentiment-analysis)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
# IMDb Sentiment Analysis - NLP Project

This project implements sentiment analysis on the IMDb movie reviews dataset using Natural Language Processing (NLP) techniques and a Logistic Regression classifier.

---

## Objective

The goal is to build a binary classification model that predicts whether a movie review is positive or negative based on its textual content.

---

## Tools and Libraries

- Python
- NLTK
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

---

## Workflow

1. Load movie review dataset from NLTK's `movie_reviews` corpus
2. Preprocess the text:
   - Remove stopwords
   - Apply stemming
3. Vectorize the reviews using TF-IDF (top 5000 features)
4. Split the data into training and testing sets
5. Train a Logistic Regression model
6. Evaluate the model using accuracy, confusion matrix, and classification report
7. Create a function to predict sentiment for custom text inputs
8. Save the trained model and vectorizer using `joblib`

---

## Model Performance

- Accuracy: Approximately 85–88% depending on random state
- Evaluation: Confusion matrix and classification report
- Supports real-time sentiment predictions on custom inputs

---

## Example Usage

```python
predict_sentiment("The movie was fantastic and emotionally powerful.")
# Output: pos

predict_sentiment("It was boring and a complete waste of time.")
# Output: neg
```
