# Fake Review Detection using Logistic Regression

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# 1. Load dataset
data = pd.read_csv("dataset.csv")

# Display dataset
print("Dataset Preview:\n")
print(data.head())

print("\nColumns in Dataset:", data.columns)

# 2. Convert labels (CG = Genuine, OR = Fake)
data['label'] = data['label'].map({'CG':0, 'OR':1})

# 3. Text Cleaning Function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning (use correct column name text_)
data['clean_review'] = data['text_'].apply(clean_text)

# 4. Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['clean_review']).toarray()
y = data['label']

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Test with custom review
def predict_review(review):
    review = clean_text(review)
    vector = vectorizer.transform([review]).toarray()
    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Fake Review")
    else:
        print("Genuine Review")

# Example
# Take input from user
user_review = input("Enter a review: ")

predict_review(user_review)