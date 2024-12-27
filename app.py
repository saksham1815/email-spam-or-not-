import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Check and download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    """Preprocess the text input."""
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    text = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords and punctuation
    text = [ps.stem(i) for i in text]  # Perform stemming
    return " ".join(text)

# Load vectorizer and model
tfidf = None
model = None

try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        logger.info("Vectorizer loaded successfully.")
except FileNotFoundError:
    st.error("Vectorizer file not found. Ensure 'vectorizer.pkl' is available in the app directory.")
    logger.error("Vectorizer file not found.")

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        logger.info("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file not found. Ensure 'model.pkl' is available in the app directory.")
    logger.error("Model file not found.")

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for user message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    input_sms = input_sms.strip()  # Remove leading/trailing whitespace
    if not input_sms:
        st.error("Please enter a valid message.")
    elif tfidf is None or model is None:
        st.error("Required resources (vectorizer/model) are not loaded. Ensure the necessary files are available.")
    else:
        try:
            # Preprocess the input message
            transformed_sms = transform_text(input_sms)
            logger.info(f"Transformed message: {transformed_sms}")

            # Vectorize the preprocessed message
            vector_input = tfidf.transform([transformed_sms])
            logger.info("Message vectorized successfully.")

            # Make prediction
            result = model.predict(vector_input)[0]
            logger.info(f"Prediction result: {'Spam' if result == 1 else 'Not Spam'}")

            # Display result
            st.header("Spam" if result == 1 else "Not Spam")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            logger.error(f"Prediction error: {e}")
