#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app title
st.title("Resume Classification App")
st.write("Classify resumes into categories based on their content.")

# Input text area for the resume details
resume_details = st.text_area("Enter Resume Details Here:")

# Button to make predictions
if st.button("Classify"):
    if resume_details.strip():
        # Preprocess and make predictions
        resume_tfidf = vectorizer.transform([resume_details])
        prediction = model.predict(resume_tfidf)

        # Display the prediction
        st.success(f"The predicted category is: *{prediction[0]}*")
    else:
        st.error("Please enter resume details to classify.")


# In[7]:


import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load the best model and TF-IDF vectorizer
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load the vectorizer used for training the model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocess the text data
stop_words = stopwords.words('english')

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Streamlit UI for input and prediction
st.title("Resume Category Classifier")

# Upload resume text
st.write("Upload a resume or paste the resume content here to classify it:")
resume_input = st.text_area("Resume Text", height=300)

# Button for prediction
if st.button('Predict Category'):
    if resume_input.strip() != "":
        # Preprocess the input text
        preprocessed_text = preprocess_text(resume_input)
        
        # Vectorize the input using the same vectorizer
        input_vect = vectorizer.transform([preprocessed_text])
        
        # Make prediction
        prediction = best_model.predict(input_vect)
        
        # Display the result
        st.write(f"The predicted category for the resume is: **{prediction[0]}**")
    else:
        st.write("Please enter the resume details for prediction.")


# In[ ]:




