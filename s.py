import numpy as np
import pickle as pkl
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB





model_filename = 'Sentiment Analysis.pkl'
vectorizer_filename ='vectorizer.pkl'


with open(model_filename, 'rb') as model_file:
    loaded_model = pkl.load(model_file)
with open(vectorizer_filename,'rb') as vectorizer_file:
    vectorizer= pkl.load(vectorizer_file)

    
st.title('Sentiment Analyser App')
st.write('This app uses Naive Bayes Algorithm to analyze the sentiment of reviews')


form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')

submit = form.form_submit_button('Submit')

if submit and user_input:

    user_input = [user_input]

    #vectorizer = TfidfVectorizer(max_features=1515)
    user_input_to_model = vectorizer.transform(user_input)

    #user_input_to_model = user_input_to_model.toarray()

    # user_input_to_model.resize(len(user_input_to_model), 1515, refcheck=False)

    

    result = loaded_model.predict(user_input_to_model)[0]

    if result == 0:
        st.error('Negative Sentiment')
    elif result == 1:
        st.warning('Neutral Sentiment')
    elif result == 2 :
        st.success('Positive Sentiment')