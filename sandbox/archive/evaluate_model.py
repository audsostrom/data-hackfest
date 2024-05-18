import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
import os

# download nltk data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# load the trained model and tokenizer
output_dir = './results'
model = RobertaForSequenceClassification.from_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(output_dir)

# load the multilabel binarizer
mlb = MultiLabelBinarizer()
mlb.classes_ = np.load(os.path.join(output_dir, 'mlb_classes.npy'), allow_pickle=True)


# function to predict themes from text
def predict_themes(synopsis, top_n=None):
    # preprocess the input text
    preprocessed_text = preprocess_text(synopsis)
    # tokenize the input text
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # get model predictions
    outputs = model(**inputs)
    logits = outputs.logits
    # apply sigmoid to get probabilities
    probs = torch.sigmoid(logits).detach().numpy().flatten()

    # associate probabilities with theme labels
    themes_with_probs = [(label, prob) for label, prob in zip(mlb.classes_, probs)]

    # sort themes by probability, highest first
    themes_with_probs.sort(key=lambda x: x[1], reverse=True)

    # return only the top N results if top_n is specified
    if top_n is not None:
        return themes_with_probs[:top_n]

    return themes_with_probs

# example usage
synopsis = ("emotions run wild in the mind of a little girl who is uprooted from her peaceful life "
            "in the midwest and forced to move to san francisco in this pixar adventure from director "
            "pete docter (up, monsters inc.). young riley was perfectly content with her life when her "
            "father landed a new job in san francisco, and the family moved across the country. now, as "
            "riley prepares to navigate a new city and attend a new school, her emotional headquarters becomes"
            " a hot bed of activity. as joy (voice of amy poehler) attempts to keep riley feeling happy "
            "and positive about the move, other emotions like fear (voice of bill hader), anger (voice "
            "of lewis black), disgust (voice of mindy kaling) and sadness (phyllis smith) make the transition "
            "a bit more complicated.")
predicted_themes = predict_themes(synopsis, 5)
print("Predicted themes:", predicted_themes)
