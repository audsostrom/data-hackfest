import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from fuzzywuzzy import fuzz, process
from pprint import pprint

# preprocess text: lowercase, remove punctuation, numbers, tokenize, remove stopwords, lemmatize, and join back to string
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# check gpu availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device} torch")

# load data
print(f"loading csv...")
raw_data = pd.read_csv("test_clean_data.csv")
print(f"........loaded")

# select and prepare data columns
data = raw_data[['kaggle_id', 'kaggle_title', 'allmovie_synopsis', 'allmovie_themes']].copy()
data['allmovie_themes'] = data['allmovie_themes'].fillna('')
data['allmovie_synopsis'] = data['allmovie_synopsis'].fillna('')

# preprocess data
print(f"preprocessing inputs...")
data['allmovie_synopsis_processed'] = data['allmovie_synopsis'].apply(preprocess_text)
data['allmovie_themes'] = data['allmovie_themes'].str.lower().str.split(', ')
print(f"....inputs preprocessed")

# load tokenizer and model
print(f"loading tokenizer and model...")
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
print(f"....tokenizer and model loaded")


# generate embeddings for batch of texts
def generate_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# define file paths
embeddings_file = "models/embeddings.npy"
similarity_file = "models/similarity_df.pkl"

# check and load embeddings and similarity matrix
if os.path.exists(embeddings_file) and os.path.exists(similarity_file):
    print(f"loading embeddings and pickle...")
    embeddings = np.load(embeddings_file)
    similarity_df = pd.read_pickle(similarity_file)
    print(f"...embeddings and pickle loaded")
else:
    # process and save embeddings and similarity matrix
    batch_size = 16
    embeddings = []

    print(f"starting embeddings processing...")
    batch = 0
    for i in range(0, len(data), batch_size):
        print(f"processing batch: {batch}")
        batch_texts = data['allmovie_synopsis_processed'].iloc[i:i + batch_size].tolist()
        batch_embeddings = generate_embeddings(batch_texts, tokenizer, model)
        # pprint(batch_texts)
        # print()
        # pprint(batch_embeddings)
        # print("-"*100)
        embeddings.append(batch_embeddings)
        batch += 1

    embeddings = np.vstack(embeddings)
    print(f".......processed embeddings")

    # one-hot encode the 'allmovie_themes' column
    mlb = MultiLabelBinarizer()
    X_themes = mlb.fit_transform(data['allmovie_themes'])

    # combine the DistilBERT embeddings and one-hot encoded themes
    X_combined = np.hstack((embeddings, X_themes))

    # calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(X_combined)

    # convert the similarity matrix to a DataFrame for lookup
    similarity_df = pd.DataFrame(similarity_matrix, index=data['kaggle_title'], columns=data['kaggle_title'])

    # save the embeddings and similarity matrix
    np.save(embeddings_file, embeddings)
    similarity_df.to_pickle(similarity_file)


# recommend movies based on similarity
def recommend_movies(movie_title, similarity_df, num_recommendations=5):
    if movie_title not in similarity_df.index:
        #print(similarity_df.columns.tolist())
        best_match, score = process.extractOne(movie_title, similarity_df.columns.tolist(), scorer=fuzz.token_sort_ratio)
        print(f"'{movie_title}' not in database, displaying results for: '{best_match}' ({score} similarity)")
        movie_title = best_match

    # get the similarity scores for the given movie
    similarity_scores = similarity_df[movie_title]

    # sort the movies by similarity score
    similar_movies = similarity_scores.sort_values(ascending=False)

    # exclude the input movie itself
    similar_movies = similar_movies.drop(movie_title)

    # get the top N recommendations
    top_recommendations = similar_movies.head(num_recommendations)

    return top_recommendations


# example usage
print()
movie_title = "Coco"
recommendations = recommend_movies(movie_title, similarity_df)
print(f"movies similar to '{movie_title}':\n", recommendations)
