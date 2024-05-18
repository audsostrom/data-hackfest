import ast
import sys

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import re
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow as tf
import string

from pprint import pprint
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Set options to display all columns
pd.set_option('display.max_columns', None)

# Set options to display more rows, e.g., 100 rows
pd.set_option('display.max_rows', 100)

# If you also want to expand the width of each column to avoid truncation of data
pd.set_option('display.max_colwidth', None)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
data = pd.read_csv("clean_data.csv")
data = data.dropna(subset=['kaggle_tagline', 'grouplens_movieId'])
# missing_data = data.isnull().mean() * 100
# print(missing_data)
# dropping:
# kaggle_tagline


def clean_text(dstring):
    translator = str.maketrans('', '', string.punctuation)
    if isinstance(dstring, str):
        # standardize text by making it lower case, replacing special characters and spaces
        # print(string)
        cleaned_string = dstring.lower()
        cleaned_string = cleaned_string.translate(translator)
        # print(cleaned_string)
        # print("-"*100)
        return cleaned_string
    return ''


def clean_list(string_list):
    if isinstance(string_list, str):
        cleaned = ast.literal_eval(string_list)
        cleaned = [item.lower() for item in cleaned]
        cleaned = ', '.join(str(item) for item in cleaned)
        return cleaned
    return ''


# define some useful helper functions
# these will remove the parenthesis from the titles
# and the pipes between genres for each item and finally
# remove the null items for movies with no genre
def remove_pars(x):
    x = str(x)
    return re.sub('[()]', "", x)


def remove_pipes(x):
    x = str(x)
    return re.sub('\|', " ", x)


def remove_nulls(a, b, i):
    string_m = a[i] + " " + b[i]
    return re.sub("\(no genres listed\)", "", string_m)


data['clean_kaggle_overview'] = data['kaggle_overview'].apply(clean_text)
data['clean_kaggle_tagline'] = data['kaggle_tagline'].apply(clean_text)
data['clean_kaggle_genres'] = data['kaggle_genres'].apply(clean_text)
data['clean_kaggle_keywords'] = data['kaggle_keywords'].apply(clean_text)
data['clean_allmovie_details_genres'] = data['allmovie_details_genres'].apply(clean_text)
data['clean_allmovie_details_sub-genres'] = data['allmovie_details_sub-genres'].apply(clean_text)
data['clean_allmovie_keywords'] = data['allmovie_keywords'].apply(clean_text)
data['clean_allmovie_themes'] = data['allmovie_themes'].apply(clean_text)
data['clean_allmovie_synopsis'] = data['allmovie_synopsis'].apply(clean_text)
data['clean_webeaver_allmovie_themes'] = data['webeaver_allmovie_themes'].apply(clean_list)
data['clean_webeaver_kaggle_themes'] = data['webeaver_kaggle_themes'].apply(clean_list)
data['input_text'] = data['clean_kaggle_tagline'].str.cat([
    data['clean_kaggle_genres'],
    data['clean_kaggle_keywords'],
    data['clean_allmovie_details_genres'],
    data['clean_allmovie_details_sub-genres'],
    data['clean_allmovie_keywords'],
    data['clean_allmovie_themes'],
    data['clean_webeaver_allmovie_themes'],
    data['clean_webeaver_kaggle_themes']
], sep=' ', na_rep='')
data['input_text'] = data['input_text'].apply(clean_text)

scaler = MinMaxScaler()
columns_to_scale = ['grouplens_urate_average', 'kaggle_vote_average']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data['grouplens_urate_count'] = np.log1p(data['grouplens_urate_count'])
data['log_kaggle_vote_count'] = np.log1p(data['kaggle_vote_count'])
scaler2 = StandardScaler()
log_columns = ['kaggle_vote_count', 'grouplens_urate_count']
data[log_columns] = scaler.fit_transform(data[log_columns])
data = data.dropna(subset=['input_text', 'grouplens_urate_average', 'kaggle_vote_average',
                           'grouplens_urate_count', 'log_kaggle_vote_count'])
# descriptive_stats = data[['grouplens_urate_average', 'grouplens_urate_count', 'kaggle_vote_average', 'kaggle_vote_count']].describe()
# print(descriptive_stats)

# make a list of the input strings for each item from these bits of data.
input_string = data['input_text']
ratings = pd.read_csv("data/grouplens/ml-25m/ratings.csv")
rating_mask = ratings['movieId'].isin(data['grouplens_movieId'])
filtered_ratings = ratings[rating_mask]
# print(ratings.shape)
# print(filtered_ratings.shape)
filtered_ratings.to_csv("data/grouplens/ml-25m/filtered_ratings.csv")


# this will be using a GPU to speed things up
# but will default to CPU if no devices are available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create embeddings for each item
embeddings_list = []
for _, i in enumerate(input_string):
    encoded_input = tokenizer(i, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    embeddings_list.append(embeddings)
    if _ % 1000 == 0:
        print(f"Processed {_} texts")

# extract the embeddings
embeddings_list_tensors = []
for i in embeddings_list:
    d = i.cpu()[0].numpy()
    embeddings_list_tensors.append(d)

# save them to local file.
text_embeddings_df = pd.DataFrame(np.vstack(embeddings_list_tensors))
# final_data = pd.concat([text_embeddings_df.reset_index(drop=True),
#                         data[['grouplens_urate_average', 'kaggle_vote_average',
#                               'grouplens_urate_count', 'log_kaggle_vote_count']].reset_index(drop=True)], axis=1)
text_embeddings_df.to_csv("embeddings/data.csv")