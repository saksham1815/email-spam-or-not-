import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set a custom NLTK data directory
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure NLTK data is downloaded and accessible
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    """Preprocess the text input."""
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    # Perform stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load the vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    tfidf = None

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for user message
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a valid message.")
    else:
        if tfidf is None or model is None:
            st.error("Model or vectorizer not loaded. Please check the setup.")
        else:
            try:
                # 1. Preprocess the input message
                transformed_sms = transform_text(input_sms)

                # 2. Vectorize the preprocessed message
                vector_input = tfidf.transform([transformed_sms])

                # 3. Make prediction
                result = model.predict(vector_input)[0]

                # 4. Display result
                if result == 1:
                    st.header("Spam")
                else:
                    st.header("Not Spam")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
