import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
nltk.download('stopwords')
nltk.download('punkt')


# Load data and model
with open('intents.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned_patterns'] = df['patterns'].apply(clean_text)

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['cleaned_patterns'])

# Load the trained model
model = tf.keras.models.load_model('model.h5')  # Ganti 'your_model_path' dengan path ke model Anda

# Function to predict tag
def get_response_lstm(input_text):
    # Cleaning text
    input_clean = clean_text(input_text)
    # Tokenize and pad input text
    input_sequence = tokenizer.texts_to_sequences([input_clean])
    input_padded = pad_sequences(input_sequence, maxlen=9, padding='post')

    # Predict using LSTM model
    prediction = model.predict(input_padded)
    predicted_index = np.argmax(prediction, axis=1)[0]

    label_encoder = LabelEncoder()
    label_encoder.fit(df['tag'])
    # Get predicted intent label using the fitted LabelEncoder instance
    intent_label = label_encoder.inverse_transform([predicted_index])[0]

    # Choose a random response corresponding to the predicted intent label
    response = random.choice([res for intent in data['intents'] if intent['tag'] == intent_label for res in intent['responses']])

    return response

# Streamlit app
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
    response = get_response_lstm(prompt)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)