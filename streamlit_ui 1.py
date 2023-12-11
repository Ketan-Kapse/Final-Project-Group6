import streamlit as st
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

st.title('GROUP 6')
model_path_db = os.path.join(os.getcwd(), 'Weights\\distilbert')
model_path_rb = os.path.join(os.getcwd(), 'Weights\\roberta')
options = ['Distilbert-Base', 'roBERTa-large', 'Linear Regression', 'Naive Bayes']

selected_option = st.selectbox('Select a Model', options)

if selected_option == 'Distilbert-Base':
    st.title('DistilBert')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(model_path_db)
    
if selected_option == 'roBERTa-large':
    st.title('roberta')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(model_path_rb)
    model.eval()
if selected_option == 'Linear Regression':
    st.title('Linear Regression')
if selected_option == 'Naive Bayes':
    st.title('Naive Bayes')



def predict_label_ml(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = logits.argmax().item()
        probabilities = torch.softmax(logits, dim=1)
        probabilities = probabilities[0].tolist()
        if predicted_label_id == 0:
            predicted_label = "The Predicted Sentiment is Negative. (ID =  " + str(predicted_label_id) + " )" + " Label Probabilites: "+ str(probabilities)
        if predicted_label_id == 1:
            predicted_label = "The Predicted Sentiment is Neutral. (ID =  " + str(predicted_label_id) + " )"+ " Label Probabilites: "+ str(probabilities)
        if predicted_label_id == 2:
            predicted_label = "The Predicted Sentiment is Positive. (ID =  " + str(predicted_label_id) + " )"+ " Label Probabilites: "+ str(probabilities)
    
    return predicted_label


def predict_label_classical(text, model_name):
    model_name = model_name
    if model_name == 'lr':
        model_path = os.path.join(os.getcwd(), 'Weights\\linearRegression\\')
        with open(model_path+'lr_model.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Loaded model successfully!") 

        with open(model_path+'vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
    
    if model_name == 'nb':
        model_path = os.path.join(os.getcwd(), 'Weights\\naiveBayes\\')
        with open(model_path+'nb_model.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Loaded model successfully!") 

        with open(model_path+'vectorizer_nb.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)


    def preprocess_text(text):
        processed_text = loaded_vectorizer.transform([text])
        return processed_text
    
    text = preprocess_text(text)
    predicted_label_id = model.predict(text)
    probabilities = model.predict_proba(text)

    if predicted_label_id == 0:
        predicted_label = "The Predicted Sentiment is Negative. (ID =  " + str(predicted_label_id[0]) + " )" + ". Label Probabilities: " + str(probabilities[0])
    if predicted_label_id == 1:
        predicted_label = "The Predicted Sentiment is Neutral. (ID =  " + str(predicted_label_id[0]) + " )" + ". Label Probabilities: " + str(probabilities[0])
    if predicted_label_id == 2:
        predicted_label = "The Predicted Sentiment is Positive. (ID =  " + str(predicted_label_id[0]) + " )"   + ". Label Probabilities: " + str(probabilities[0])

    return predicted_label
    





st.header('Text Prediction')
input_text = st.text_input('Enter text for prediction:',)
if st.button('Predict'):
    if selected_option == "Distilbert-Base" or selected_option =="roBERTa-large":
        prediction = predict_label_ml(input_text)
        st.write('Predicted Label:', prediction)
    if selected_option == "Linear Regression":
        prediction = predict_label_classical(input_text, 'lr')
        st.write('Predicted Label:', prediction)
    if selected_option == "Naive Bayes":
        prediction = predict_label_classical(input_text, 'nb')
        st.write('Predicted Label:', prediction)
    