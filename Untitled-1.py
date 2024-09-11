
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore

# Sample training data
documents = [
    "I love coding",
    "Python is a great programming language",
    "Machine learning is fascinating",
    "I enjoy building AI models"
]
labels = ["coding", "programming", "machine learning", "AI"]

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Test the classifier
test_document = "I want to learn Python"
test_X = vectorizer.transform([test_document])
predicted_label = classifier.predict(test_X)

print("Predicted label:", predicted_label)