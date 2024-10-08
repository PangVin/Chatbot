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
nltk.download('punkt_tab')


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
model = tf.keras.models.load_model('model.h5') 

# Function untuk memprediksi pakai model LSTM
def get_response_lstm(input_text):
    # Hapus Text dari function clean_text yang diatas
    input_clean = clean_text(input_text)
    print(f"Cleaned input: {input_clean}")
    # Tokenize and pad input text
    # Mengubah teks yang telah dibersihkan menjadi urutan token menggunakan tokenizer yang telah dilatih
    input_sequence = tokenizer.texts_to_sequences([input_clean])
    print(f"Tokenized input: {input_sequence}")
    # Memastikan urutan token memiliki panjang tetap dengan menambahkan padding di akhir jika diperlukan
    input_padded = pad_sequences(input_sequence, maxlen=9, padding='post')
    print(f"Padded input: {input_padded}")

    # Predict using LSTM model
    # Menggunakan model LSTM untuk memprediksi intent dari input yang telah diproses
    prediction = model.predict(input_padded)
    print(f"Model prediction: {prediction}")
    # Menentukan indeks label intent dengan probabilitas tertinggi dari hasil prediksi model LSTM
    predicted_index = np.argmax(prediction, axis=1)[0]
    print(f"Predicted index: {predicted_index}")
    # Membuat instance LabelEncoder baru dan melatihnya dengan label intent dari dataframe df
    label_encoder = LabelEncoder()
    label_encoder.fit(df['tag'])
    # Get predicted intent label using the fitted LabelEncoder instance
    # Mengubah indeks label intent yang diprediksi kembali menjadi label intent yang dapat dibaca
    intent_label = label_encoder.inverse_transform([predicted_index])[0]
    print(f"Intent label: {intent_label}")

    # Choose a random response corresponding to the predicted intent label
    # Memilih respons secara acak dari daftar respons yang sesuai dengan label intent ya ng diprediksi
    response = random.choice([res for intent in data['intents'] if intent['tag'] == intent_label for res in intent['responses']])
    print(f"Chosen response: {response}")
    return response

# Function untuk rule based method
def rule_based_response(prompt):
    responses = {
        "hello": "Hi there! How can I help you today?",
        "bye": "Goodbye! Have a great day!",
        "how are you": "I'm just a bot, but I'm doing well. How can I assist you?",
        "thank you": "You're welcome! If you have any more questions, feel free to ask.",
        "feeling sad": "I'm sorry to hear that you're feeling this way. Sometimes talking about it can help. Would you like to chat more about it?",
        "stressed": "Stress can be overwhelming. It's important to take breaks and find ways to relax. Would you like some tips on managing stress?",
        "happy": "That's great to hear! If you'd like to share more about what's making you happy, I'm here to listen.",
        "lonely": "Feeling lonely can be tough. Remember, reaching out to friends or loved ones can make a difference. If you need to talk, I'm here for you.",
        "help": "I'm here to help. If you're struggling, please consider talking to a mental health professional or someone you trust.",
        "can you help me?": "I'm here to help. If you're struggling, please consider talking to a mental health professional or someone you trust." 
    }
    # Menyederhanakan prompt untuk mencocokkan dengan aturan
    prompt = prompt.lower().strip()
    return responses.get(prompt, None)  # Kembalikan None jika tidak ada respons yang cocok



import streamlit as st
import random

# Greeting and recommended questions
greeting = "Hi! Welcome to the Mental Health Chatbot. How can I assist you today?"
recommended_questions = [
    "What should I do if I feel stressed?",
    "How can I handle burnout at work?",
    "Does taking short breaks help with productivity?",
]

# Daily mental health tips
mental_health_tips = [
    "Take a few minutes to breathe deeply and relax.",
    "Try to step away from work for a short walk.",
    "Write down something you're grateful for today.",
    "Reach out to a friend or loved one for a chat.",
    "Take a break and enjoy a cup of tea or coffee.",
    "Practice mindfulness or meditation for a few minutes.",
    "Listen to your favorite music to boost your mood.",
    "Spend time in nature or simply sit in a quiet space.",
    "Do something creative like drawing, writing, or crafting.",
    "Limit your screen time and focus on offline activities."
]

daily_tip = random.choice(mental_health_tips)

st.title("Mental Health Chatbot")

# Ini untuk chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    st.session_state.recommendations_shown = False  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
 # New Chat Button
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.recommendations_shown = False
        st.session_state.chat_history.append({
            "date": st.session_state.messages,
            "messages": st.session_state.messages.copy()
        })
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.experimental_rerun()  # Rerun the app to reset chat


    st.header("Your Mental Health Tools")

    # Daily Mental Health Tips
    st.subheader("Daily Mental Health Tip")
    st.info(daily_tip)


    # Feedback
    with st.expander("We value your feedback"):
        feedback = st.text_area("Please share any feedback or suggestions you have (optional):")
        if st.button("Submit Feedback", key="feedback_button"):
            st.session_state.feedback = feedback
            st.success("Thank you for your feedback!")

    # Download Chat Transcript
    with st.expander("Download Chat Transcript"):
        transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name="chat_transcript.txt",
            mime="text/plain"
        )

    # Chat History
    st.subheader("Chat History")
    for idx, chat in enumerate(st.session_state.chat_history):
        if st.button(f"Chat {idx + 1}"):
            st.session_state.messages = chat["messages"]
            st.experimental_rerun()  # Rerun the app to load selected chat

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display recommended questions if not shown yet
if not st.session_state.recommendations_shown:
    st.markdown("### Recommended Questions:")
    for question in recommended_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            response = get_response_lstm(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.recommendations_shown = True  # Hide recommendations after one is selected
            st.experimental_rerun()  # Rerun the app to update UI

# Accept user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cek apakah respons bisa didapat dari metode rule-based
    response = rule_based_response(prompt)  
    
    if response is None:
        # Jika tidak ada respons dari rule-based, gunakan model LSTM
        response = get_response_lstm(prompt)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

