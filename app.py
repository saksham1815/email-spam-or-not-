import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Ensure NLTK data is downloaded and accessible
# try:
nltk.data.find('tokenizers/punkt')
# except LookupError:
nltk.download('punkt')  # Download punkt tokenizer data

# try:
nltk.data.find('corpora/stopwords')
# except LookupError:
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    """Preprocess the text input."""
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

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
