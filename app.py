import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import nltk
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
import re
import random
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from googletrans import Translator, LANGUAGES

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
    text = re.sub(r'\s+', ' ', text).strip()
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

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='id', dest='en')
    return translation.text

def translate_to_indonesian(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='id')
    return translation.text

def detect_language(text):
    translator = Translator()
    try:
        detected = translator.detect(text)
        if detected and hasattr(detected, 'lang'):
            return detected.lang
        else:
            st.warning("Failed to detect language. Defaulting to English.")
            return 'en'  # Default to English
    except AttributeError as e:
        st.warning(f"Error detecting language: AttributeError - {e}")
        return 'en'  # Default to English in case of AttributeError
    except Exception as e:
        st.warning(f"Error detecting language: {type(e).__name__} - {e}")
        return 'en'  # Default to English for any other exception


def get_bot_response(user_input):
        # Detect user input language
    input_lang = detect_language(user_input)

    if input_lang == "id":
        user_input = translate_to_english(user_input)

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


    # Translate responses if necessary
    if input_lang == 'en':
        translated_response = bot_response
    elif input_lang == 'id':
        translated_response = translate_to_indonesian(bot_response)
    else:
        translated_response = bot_response  # fallback to original response

    return translated_response

# Streamlit App
st.title("Chatbot Mental Health")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    response = get_bot_response(prompt)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

