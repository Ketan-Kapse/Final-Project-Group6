import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

model_path_db = os.getcwd()
model_path_rb = os.getcwd()+'//roberta'
options = ['Model 1', 'Model 2']

selected_option = st.selectbox('Select a Model', options)

if selected_option == 'Model 1':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(model_path_db)
if selected_option == 'Model 2':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(model_path_rb)
model.eval()
def predict_label(text):
    # Tokenize text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = logits.argmax().item()
        if predicted_label_id == 0:
            predicted_label = "The Predicted Sentiment is Negative. (ID =  " + str(predicted_label_id) + " )"
        if predicted_label_id == 1:
            predicted_label = "The Predicted Sentiment is Neutral. (ID =  " + str(predicted_label_id) + " )"
        else:
            predicted_label_id = "The Predicted Sentiment is Positive. (ID =  " + str(predicted_label_id) + " )"
    
    return predicted_label

st.title('Model Deployment')

st.header('Text Prediction')
input_text = st.text_input('Enter text for prediction:', 'Type here...')
if st.button('Predict'):
    prediction = predict_label(input_text)
    st.write('Predicted Label:', prediction)