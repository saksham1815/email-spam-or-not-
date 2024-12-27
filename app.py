import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for user message
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a valid message.")
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
