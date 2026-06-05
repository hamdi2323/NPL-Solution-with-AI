"""
===============================================================
Python script to demonstrate a document classification and topic modelling
===============================================================
This program demonstrate two natural language process tasks i.e. Document Classification
(Supervised Learning)
Part 1: Document Classification (Supervised Learning)
        -TF-IDF Vectorization
        -Train/Test split
        -Multinomial Naive Bayes Classification
        -Accuracy Evaluation
        -Classification Report
        -Interaction Predictions

Part 2: Topic Modelling (Unsupervised Learning)
        -TF-IDF Vectorization
        -Latent Dirichlet Allocation (LDA)
        -Topic Discovery
        -Topic Interpretation
        -Dominant Topic Assignment
Dataset location: files/articles.json, files/topics.json

Requirements:
    pip install scikit-learn
Author: Xamdi Salaad
Date: 04-06-2026
"""
#------------------------------------------------------------
#0.Import required modules
#-----------------------------------------------------------
import json,re,sys
import numpy as np
import pandas as pd

from pathlib import Path  # Fixed: Changed from Pathlib to pathlib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import warnings
#Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")
#------------------------------------------------------------
#1.Helper function (output formatting)
#-----------------------------------------------------------
def print_seperator(title: str = "") -> None:  # Fixed: Added parameter
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)

#------------------------------------------------------------
#Part 1: Document classification (Supervised learning)
#-----------------------------------------------------------
print_seperator("Part 1: Document Classification (Supervised Learning)")

#------------------------------------------------------------
#Step I:Load the news articles dataset
#-----------------------------------------------------------
print("\nLoad the news articles dataset...")

articles_files = Path("../files/articles.json")
with open(articles_files, "r") as file:
    articles_data = json.load(file)

    articles_df = pd.DataFrame(articles_data)

    print(f"New articles loaded successfully. "
          f"\nNumber of articles: {len(articles_df)}")

    print(f"\nAvailable categories:"
          f" {sorted(articles_df['category'].unique())}")  # Fixed: Changed columns to indexing

#------------------------------------------------------------
#Step II:Combine title and content
#-----------------------------------------------------------
print("\nCombining title and content fields....")

articles_df['text'] = (
        articles_df['title'].fillna(" ") + " "
        + articles_df['content'].fillna(" ")
        )
X_text = articles_df['text']
y = articles_df['category']

print("Text preparation complete...")
#------------------------------------------------------------
#Step III: Convert text into TF-IDF features
#-----------------------------------------------------------
print("\nCreating TF-IDF features...")

tfidf_classifier = TfidfVectorizer(
    stop_words='english',
    max_features=3000,
)

X_features = tfidf_classifier.fit_transform(X_text)

print(f"Number of documents: {X_features.shape[0]}"
      f"Number of TF-IDF features: {X_features.shape[1]}")
#------------------------------------------------------------
#Step IV: Train/Test split
#-----------------------------------------------------------
print("\nSplitting the dataset into training and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.20,
    random_state=42, #for reproducibility
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}"
      f"\nTesting samples: {X_test.shape[0]}")

#------------------------------------------------------------
#Step V: Train the classifier
#-----------------------------------------------------------

print("\nTraining Multinomial Naive Bayes Classifier...")

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

print("Training complete")

#------------------------------------------------------------
#Step VI: Evaluate the model
#-----------------------------------------------------------
print("Evaluating classifier...")
y_predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
print_seperator("Classification Result")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report")
print(classification_report(y_test, y_predictions))

#------------------------------------------------------------
#2.Example Predictions
#-----------------------------------------------------------
print_seperator("EXAMPLE PREDICTION")

example_articles = [
    "The government has introduced new tax reforms and budget policies.",
    "The football team secured a dramatic victory in the championship final",
    "A new artificial intelligence platform has been launched by technology company"  # Fixed: Changed 'technolgy' to 'technology'
]

for text in example_articles:
    vectorised_text = tfidf_classifier.transform([text])
    predicted_category = classifier.predict(vectorised_text)

    print(f"\nText: {text}")
    print(f"Predicted category: {predicted_category[0]}")  # Fixed: Added [0] to get the value
#------------------------------------------------------------
#3.Interactive prediction
#-----------------------------------------------------------
print_seperator("INTERACTIVE NEWS CLASSIFICATION")

print("Enter a news headline or short articles")
print("Press ENTER on an empty line to continue to Topic Modelling.\n")
while True:
    user_text = input("News or Article text: ").strip()

    if user_text == "":
        break

    user_feature = tfidf_classifier.transform([user_text])
    prediction = classifier.predict(user_feature)[0]

    print(f"Predicted category: {prediction}\n")

#------------------------------------------------------------
#Part 2: Topic Modelling (unsupervised learning)
#-----------------------------------------------------------
print_seperator("Part 2: Topic Modelling (Unsupervised Learning)")

#------------------------------------------------------------
#Step I: Load movie reviews dataset
#-----------------------------------------------------------

print("\nLoading the movies reviews dataset...")
topic_files = Path("../files/topics.json")

with open(topic_files, "r") as file:
    topics_data = json.load(file)
    reviews_df = pd.DataFrame(topics_data)

    print(f"Movies reviews loaded successfully. "
          f"\nNumber of reviews: {len(reviews_df)}")

#------------------------------------------------------------
#Step II: Extract review text
#-----------------------------------------------------------
print("\nPreparing review texts...")

reviews_texts = reviews_df['review_text'].fillna(" ")

#------------------------------------------------------------
#Step III: TF-IDF Vectors
#-----------------------------------------------------------
print("\nCreating TF-IDF matrix for topic modelling...")
tfidf_topic = TfidfVectorizer(
    stop_words='english',
    max_features=2000,
)
review_features = tfidf_topic.fit_transform(reviews_texts)

print(f"Number of documents:{review_features.shape[0]}")
print(f"Number of features: {review_features.shape[1]}")  # Fixed: Added missing f-string prefix

#------------------------------------------------------------
#Step IV: Apply Latent Dirichlet Allocation (LDA)
#-----------------------------------------------------------
print("\nApplying Latent Dirichlet Allocation...")

number_of_topics = 5
lda_model = LatentDirichletAllocation(
    n_components=number_of_topics,
    random_state=42,
    learning_method='batch',
)

lda_model.fit(review_features)

print("Topic Modelling complete...")

#------------------------------------------------------------
#Step V: Display top words per topic
#-----------------------------------------------------------
print_seperator("DISCOVERED TOPICS")
feature_names = tfidf_topic.get_feature_names_out()

for topic_index, topic in enumerate(lda_model.components_):  # Fixed: Changed 'top' to 'topic'
    top_word_indices = topic.argsort()[-10:][::-1]  # Fixed: Changed order of slicing
    top_words = [feature_names[i] for i in top_word_indices]
    print(f"\nTopic {topic_index +1}")
    print("-"*40)
    print(",".join(top_words))

#------------------------------------------------------------
#Step VI: Assign Dominant topic to each review
#-----------------------------------------------------------
print_seperator("ASSIGN DOMINANT TOPICS")

topic_probabilities = lda_model.transform(review_features)

dominant_topics = np.argmax(topic_probabilities, axis=1)  # Fixed: Changed 'argmac' to 'argmax'
reviews_df['dominant_topic'] = dominant_topics  # Fixed: Changed 'dominant_topics' to 'dominant_topic' for consistency

print("Topic assignment complete...")
topic_counts = reviews_df['dominant_topic'].value_counts().sort_index()

print("\nNumber of reviews per topic:")
for topic_id, count in topic_counts.items():
    print(f"Topic {topic_id + 1}: {count}")

#------------------------------------------------------------
#Step VII: Display sample reviews per topic
#-----------------------------------------------------------
print_seperator("SAMPLE REVIEW BY TOPIC")

for topic_number in range(number_of_topics):
    print(f"\nTopic {topic_number + 1}")
    print("-"*60)

    topic_reviews = reviews_df[
        (reviews_df['dominant_topic'] == topic_number)
    ]
    samples = topic_reviews.head(2)

    if len(samples) == 0:
        print("No reviews assigned to this topic.")
        continue

    for _, row in samples.iterrows():  # Fixed: Changed 'in sample.iterrows()' to '_, row in samples.iterrows()'
        review_text = str(row['review_text'])  # Fixed: Changed 'reviews_texts' to 'review_text'

        preview = review_text[:200].replace("\n", " ")  # Fixed: Changed 'reviews_texts' to 'review_text'

        print(f"\nMovie: {row.get('movie_title', 'unknown')}")
        print(f"Review preview: {preview}...")

print_seperator("END OF DEMONSTRATION")