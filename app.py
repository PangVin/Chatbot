import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import nltk
nltk.download('punkt')
nltk.download("stopwords")
import re
import random
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Preprocess data as before
patterns = []
responses = []
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        labels.append(intent['tag'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

cleaned_patterns = [clean_text(pattern) for pattern in patterns]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
X = vectorizer.fit_transform(cleaned_patterns)

# Convert labels to numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Function to predict response
def get_bot_response(user_input):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    predicted_label = rf_classifier.predict(input_vector)[0]

    # Get responses for the predicted intent tag
    for intent in intents['intents']:
        if intent['tag'] == encoder.inverse_transform([predicted_label])[0]:
            responses = intent['responses']
            break

    # Choose a random response from the list of responses
    bot_response = random.choice(responses)
    return bot_response

# Streamlit App
def main():
    st.title('Simple Chatbot with Streamlit')

    # User input box
    user_input = st.text_input('Enter your message:')
    if st.button('Send'):
        if user_input:
            bot_response = get_bot_response(user_input)
            st.text('Bot response:')
            st.text(bot_response)

if __name__ == '__main__':
    main()
