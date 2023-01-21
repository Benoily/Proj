#!/usr/bin/env python3.8.15
# coding: utf-8

# In[1]:

pip install python==3.8.15

import streamlit as st
from keras.models import load_model 
import spacy


# chargement et  mise en cache du modèle
@st.experimental_singleton
def load_the_model():
    
    #STATIC_FOLDER = 'C:/Users/folyb/Documents/IA/P7/ESSAI/'
    #MODEL_FOLDER = STATIC_FOLDER 
    
    global model
    #model = load_model(MODEL_FOLDER + 'bidirectional_lstm_with_return_sequences_on_embedded_heroku')
    model = load_model('./model')

    return model

my_model = load_the_model() 


# fonction qui renvoie la prédiction
def predict(tweet):
    # prédiction
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(tweet)
    tokens = [token.lemma_ for token in tokens if str(token) != ' ']
    prediction = my_model.predict(tokens)

    return prediction


st.title("Prédiction de sentiment")
st.markdown("Nous utilisons un tweet en entrée pour prédire son sentiment positif ou négatif")

# récupération du tweet
st.subheader("Entrez un tweet")
st.write('Le tweet doit être en anglais.')
tweet = st.text_input('',0,280)

# affichage de la prédiction
st.subheader("Sentiment")
if st.button("Prédire le sentiment"):
    sentiment = predict(tweet)
    if sentiment[0][0] >= 0.5:
        st.success('Le sentiment du tweet est positif')
    else:
        st.warning('Le sentiment du tweet est négatif')

