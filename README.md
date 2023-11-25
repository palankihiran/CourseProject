# CourseProject
# Sentiment Analysis README

## Project Overview

In this project, I have developed a sentiment analysis model for restaurant reviews using machine learning. I utilized restaurant reviews from both TripAdvisor and Google to build the model. The purpose of this project is to categorize restaurant reviews into three sentiment categories: Negative, Neutral, and Positive. While restaurant ratings provide a high-level summary, sentiment analysis allows us to gain deeper insights into customer feedback by understanding the sentiment behind the ratings.

## Libraries Used

The following Python libraries were used in this project:

- **Pandas:** For reading and handling data.
- **NumPy:** For data manipulation and related operations.
- **Matplotlib:** For creating plots and visualizations.
- **Seaborn:** For enhanced control over data visualization.
- **Scikit-Learn:** For Machine Learning

## Data Sources

I used two datasets for this project:

1. **TripAdvisor Data:** Contains restaurant reviews from TripAdvisor, including information such as author, title, review, rating, dates, restaurant, and location.

2. **Google Review Data:** Contains restaurant reviews from Google, including information such as author, rating, review, restaurant, location, and review length.

## Data Preprocessing

Before building the sentiment analysis model, I performed data preprocessing, which included the following steps:

- Checked for missing values in both datasets and removed rows with missing data.
- Calculated summary statistics for ratings in both datasets.

## Data Visualization

I visualized the data to gain a better understanding and discover meaningful insights:

- Plotted the distribution of ratings for TripAdvisor and Google Review.
- Analyzed the distribution of ratings by location for TripAdvisor and Google Reviews.
- Generated word clouds for TripAdvisor and Google reviews.
- Calculated and visualized the distribution of review lengths.

## Sentiment Analysis

The sentiment analysis process involves the following steps:

1. Merged the TripAdvisor and Google Review datasets based on common columns to create a combined dataset.

2. Split the ratings into three sentiment categories: Negative, Neutral, and Positive, based on predefined bins.

3. Split the combined dataset into a training set (80%) and a testing set (20%).

4. Trained a Multinomial Logistic Regression model using TF-IDF vectorization.

5. Evaluated the model's performance.

## Model Deployment

I saved the trained TF-IDF vectorizer and the sentiment analysis model using the pickle library. This allows you to deploy the model and make predictions on new restaurant reviews.

## How to Make Predictions Using the Model

To make predictions on new restaurant reviews, follow these steps:

1. Import the required libraries:

```python
import pickle
```

2. Load the TF-IDF vectorizer and sentiment analysis model using pickle:

```python
with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
```

3. Define a function to make predictions for a given restaurant review:

```python
def predict_rating_category(user_input, model=None, tfidf_vectorizer=None):
    """
    Predicts the rating category (Negative, Neutral, or Positive) for a given user input.

    Parameters:
    - model: A trained classification model (e.g., Multinomial Logistic Regression).
    - tfidf_vectorizer: A TF-IDF vectorizer fitted on the training data.
    - user_input: The user's restaurant review text.

    Returns:
    - predicted_category: The predicted rating category.
    """
    if model is None:
        model = pickle.load(open("sentiment_model.pkl", 'rb'))
        
    if tfidf_vectorizer is None:
        tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

    # Preprocess the user input using the same TF-IDF vectorizer
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Predict the rating category using the trained model
    predicted_class = model.predict(user_input_tfidf)
    probabilities = model.predict_proba(user_input_tfidf)[0]

    # Create a dictionary of probabilities for each category
    rating_categories = ['Negative', 'Neutral', 'Positive']
    probabilities_dict = {category: prob for category, prob in zip(rating_categories, probabilities)}

    return predicted_class[0], probabilities_dict
```

4. Use the function to predict the sentiment category for a restaurant review:

```python
user_input = "The food was terrible, and the service was slow."
predicted_category, probabilities = predict_rating_category(user_input, model, tfidf_vectorizer)
print(f"Predicted Rating Category: {predicted_category}")
print("Probabilities:")
for category, prob in probabilities.items():
    print(f"{category}: {prob:.4f}")
```

5. You can replace `user_input` with your own restaurant review text to make predictions.

This concludes the setup and usage of the sentiment analysis model for restaurant reviews. Feel free to use this model to analyze the sentiment of new reviews and gain insights into customer feedback.
