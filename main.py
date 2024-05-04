from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np


dataset = {
    'Greet': ["Hi", "How are you?", "Hello"],
    'Farewell': ["Goodbye", "See you later", "Take care"],
    'Inquiry': ["What's the weather like today?", "Can you tell me the time?", "Where is the nearest restaurant?"],
    'Apology':["Sorry", "My bad", "My apologies"],
    'Thank': ["Thank you", "Many Thanks", "I appreciate"],
    'Condemnation': ["You fool", "Get out of here", "You stupid"]
}

# Preprocess data
X_train = []
y_train = []
for intent, labels in dataset.items():
    X_train.extend(labels)
    y_train.extend([intent] * len(labels))

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)


def classify_intent(user_input):
    confidence = np.max(model.predict_proba([user_input]))
    predicted_intent = model.predict([user_input])[0]
    return confidence, predicted_intent


def fallback(confidence):
    if confidence < 0.28:
        return True
    else:
        return False


# Loop to allow user input and intent classification
for i in range(10):  # Change the range as needed
    user_input = input("Enter a text: ")
    confidence, predicted_intent = classify_intent(user_input)
    if(fallback(confidence)):
        print("NLU fallback: Intent could not be confidently determined")
    else:
        print(f"Predicted Intent: {predicted_intent}, Confidence: {confidence:.2f}")

