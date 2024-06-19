import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to get BERT embeddings
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to train and evaluate models
def train_and_evaluate(df):
    # Tokenize and pad sequences for LSTM
    tokenizer_lstm = Tokenizer(num_words=5000)
    tokenizer_lstm.fit_on_texts(df['cleaned_text'])
    X_lstm = tokenizer_lstm.texts_to_sequences(df['cleaned_text'])
    X_lstm = pad_sequences(X_lstm, maxlen=512)
    
    # Train-test split for LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, df['label'], test_size=0.2, random_state=42)

    # LSTM Model definition
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim=5000, output_dim=128, input_length=512))
    model_lstm.add(SpatialDropout1D(0.2))
    model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the LSTM model
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), verbose=2)

    # Evaluate LSTM model
    y_pred_lstm = (model_lstm.predict(X_test_lstm) > 0.5).astype("int32")
    accuracy_lstm = accuracy_score(y_test_lstm, y_pred_lstm)
    precision_lstm = precision_score(y_test_lstm, y_pred_lstm)
    recall_lstm = recall_score(y_test_lstm, y_pred_lstm)
    f1_lstm = f1_score(y_test_lstm, y_pred_lstm)
    
    # Combine BERT and RoBERTa embeddings
    combined_features = np.hstack((np.vstack(df['bert_embeddings']), np.vstack(df['roberta_embeddings'])))

    # Train-test split for combined features
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
        combined_features, df['label'], test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model_combined = LogisticRegression(max_iter=1000)
    model_combined.fit(X_train_combined, y_train_combined)

    # Evaluate combined features model
    y_pred_combined = model_combined.predict(X_test_combined)
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
    precision_combined = precision_score(y_test_combined, y_pred_combined)
    recall_combined = recall_score(y_test_combined, y_pred_combined)
    f1_combined = f1_score(y_test_combined, y_pred_combined)

    # Return evaluation metrics and trained models
    return {
        'LSTM': {
            'model': model_lstm,
            'Accuracy': accuracy_lstm,
            'Precision': precision_lstm,
            'Recall': recall_lstm,
            'F1 Score': f1_lstm
        },
        'Combined': {
            'model': model_combined,
            'Accuracy': accuracy_combined,
            'Precision': precision_combined,
            'Recall': recall_combined,
            'F1 Score': f1_combined
        }
    }

# Streamlit app
st.title('AI vs Human Text Classification')
st.write('Upload a CSV file with a "Text" column containing essays.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Preprocess the text
    df['cleaned_text'] = df['Text'].apply(preprocess_text)

    # Add labels for demonstration (0 for human, 1 for AI-generated)
    half_length = len(df) // 2
    df['label'] = [0] * half_length + [1] * (len(df) - half_length)
    
    # Load BERT and RoBERTa models and tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # Extract embeddings
    df['bert_embeddings'] = df['cleaned_text'].apply(lambda x: get_bert_embeddings(x, bert_tokenizer, bert_model).flatten())
    df['roberta_embeddings'] = df['cleaned_text'].apply(lambda x: get_bert_embeddings(x, roberta_tokenizer, roberta_model).flatten())

    # Train and evaluate models
    metrics = train_and_evaluate(df)

    # Display evaluation metrics
    st.write('### Model Evaluation Metrics')

    st.write('#### LSTM Model')
    st.write(f"Accuracy: {metrics['LSTM']['Accuracy']:.4f}")
    st.write(f"Precision: {metrics['LSTM']['Precision']:.4f}")
    st.write(f"Recall: {metrics['LSTM']['Recall']:.4f}")
    st.write(f"F1 Score: {metrics['LSTM']['F1 Score']:.4f}")

    st.write('#### Combined Features Model')
    st.write(f"Accuracy: {metrics['Combined']['Accuracy']:.4f}")
    st.write(f"Precision: {metrics['Combined']['Precision']:.4f}")
    st.write(f"Recall: {metrics['Combined']['Recall']:.4f}")
    st.write(f"F1 Score: {metrics['Combined']['F1 Score']:.4f}")

    # Display Predictions
    st.write('### Predictions')
    st.write('The updated CSV with predictions has been saved as `output_with_predictions.csv`.')

    # Assuming combined features model performs better, apply it to the entire dataset
    combined_features_full = np.hstack((np.vstack(df['bert_embeddings']), np.vstack(df['roberta_embeddings'])))
    df['predicted_label_combined'] = metrics['Combined']['model'].predict(combined_features_full)

    # Apply LSTM predictions
    tokenizer_lstm = Tokenizer(num_words=5000)
    tokenizer_lstm.fit_on_texts(df['cleaned_text'])
    X_full = pad_sequences(tokenizer_lstm.texts_to_sequences(df['cleaned_text']), maxlen=512)
    df['predicted_label_lstm'] = (metrics['LSTM']['model'].predict(X_full) > 0.5).astype("int32")

    st.table(df[['Text', 'predicted_label_combined', 'predicted_label_lstm']])

    # Save the DataFrame with predictions
    df.to_csv('output_with_predictions.csv', index=False)

