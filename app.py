from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert labels
data['label'] = data['label'].map({'CG':0, 'OR':1})

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Clean reviews
data['clean_review'] = data['text_'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_review']).toarray()
y = data['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction function
def predict_review(review):
    review = clean_text(review)
    vector = vectorizer.transform([review]).toarray()
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Fake Review"
    else:
        return "Genuine Review"

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    result = predict_review(review)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)