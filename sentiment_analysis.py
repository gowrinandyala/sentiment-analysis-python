import pandas as pd

df = pd.read_csv("reviews.csv")

print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# 1. Load dataset
data = pd.read_csv("reviews.csv")

print("Dataset:")
print(data)


# 2. Separate input and output
X = data["review"]
y = data["sentiment"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Predict new customer reviews
while True:

    review = input("\nEnter a customer review (or type 'exit'): ")

    if review.lower() == "exit":
        break

    prediction = model.predict([review])

    print("Predicted Sentiment:", prediction[0])

