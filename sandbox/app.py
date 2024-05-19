
import io
import json
import os

from flask import Flask, jsonify, request

# Import the necessary classes and methods from your recommender system
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np
from annoy import AnnoyIndex
import re
# Import flask and datetime module for showing date and time
import google.generativeai as genai
from flask_cors import CORS
import datetime

# Initializing flask app
app = Flask(__name__)
CORS(app)

api_key = 'AIzaSyC8WizRY4zsJsqxpC1S9bUZY25yqoZEuOk'
genai.configure(api_key=api_key)

def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'
    matches = re.finditer(pattern, text_response)
    json_objects = []
    for match in matches:
        json_str = match.group(0)
        print(json_str)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            json_obj = json.loads(extended_json_str)
            json_objects.append(json_obj)
    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer
def extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]

# Define the Recommender Network (same as your previous definition)
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_input, movie_input):
        user_vec = self.user_embedding(user_input)
        user_bias = self.user_bias(user_input)
        movie_vec = self.movie_embedding(movie_input)
        movie_bias = self.movie_bias(movie_input)
        dot = (user_vec * movie_vec).sum(1)
        return torch.sigmoid(dot + user_bias.squeeze() + movie_bias.squeeze())


# Define the personalized searcher (same as your previous definition)
class personalisedSearcher:
    def __init__(self):
        self.movies = pd.read_csv("data/grouplens/ml-25m/movies.csv")
        self.ratings = pd.read_csv("data/grouplens/ml-25m/filtered_ratings.csv")
        self.embeddings = pd.read_csv("data/embeddings/data.csv", index_col=0)
        self.item_tensor = torch.tensor(self.embeddings.values, dtype=torch.float32)

        # Initialize Annoy Index
        self.index = AnnoyIndex(self.embeddings.shape[1], 'angular')
        for i in range(len(self.item_tensor)):
            self.index.add_item(i, self.item_tensor[i].numpy())
        self.index.build(10)  # 10 trees

        self.model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

        # Define the recommendation model and load the weights
        num_users = self.ratings['userId'].nunique()
        num_movies = self.ratings['movieId'].nunique()
        embedding_size = 128  # Should match the embedding size used in training

        self.recommender = RecommenderNet(num_users, num_movies, embedding_size)
        self.recommender.load_state_dict(torch.load('CF/CF.pth', map_location=torch.device('cpu')))
        self.recommender.eval()  # Set the model to evaluation mode

    def get_user_encodings(self):
        user_ids = self.ratings["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}

        return user2user_encoded, userencoded2user

    def get_movie_encodings(self):
        movie_ids = self.ratings["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

        return movie2movie_encoded, movie_encoded2movie

    def update_ratings(self):
        user2user_encoded, _ = self.get_user_encodings()
        movie2movie_encoded, _ = self.get_movie_encodings()
        self.ratings["user"] = self.ratings["userId"].map(user2user_encoded)
        self.ratings["movie"] = self.ratings["movieId"].map(movie2movie_encoded)

        return self.ratings

    def get_user_history(self, user_id):
        df = self.update_ratings()
        watched_movies = df[df.userId == user_id]
        return watched_movies

    def get_candidate_movies(self, query):
        encoded_input = self.tokenizer(query,
                                       padding=True,
                                       truncation=True,
                                       max_length=64,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        query_embeddings = model_output.pooler_output
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        # Query Annoy Index
        nearest_ids = self.index.get_nns_by_vector(query_embeddings[0].numpy(), 10)  # Find 5 nearest neighbors

        return self.movies.iloc[nearest_ids]

    def filter_candidates(self, user_id, query):
        movies_watched_by_user = self.ratings[self.ratings.userId == user_id]
        candidates = self.get_candidate_movies(query)
        movies_not_watched = candidates[
            ~candidates["movieId"].isin(movies_watched_by_user.movieId.values)
        ]["movieId"]
        movie2movie_encoded, _ = self.get_movie_encodings()
        movies_not_watched = list(set(movies_not_watched).
                                  intersection(set(movie2movie_encoded.keys())))
        movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
        user2user_encoded, _ = self.get_user_encodings()
        user_encoder = user2user_encoded.get(user_id)
        movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

        return movie_array, movies_not_watched, movies_watched_by_user

    def personalised_search(self, user_id, query):
        movie_array, movies_not_watched, movies_watched_by_user = self.filter_candidates(user_id, query)

        # Prepare inputs for the recommender model
        user_tensor = torch.tensor([user_id] * len(movies_not_watched), dtype=torch.long)
        movie_tensor = torch.tensor(movies_not_watched, dtype=torch.long).squeeze()

        with torch.no_grad():
            scored_items = self.recommender(user_tensor, movie_tensor).numpy()

        top_rated = scored_items.argsort()[-10:][::-1]
        _, movie_encoded2movie = self.get_movie_encodings()
        recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_rated]

        return recommended_movie_ids, movies_watched_by_user

    def print_recs(self, user_id, query):
        recommendations, movies_watched_by_user = self.personalised_search(user_id, query)

        print("Showing recommendations for user: {}".format(user_id))
        print("====" * 9)
        print("Movies with high ratings from user")
        print("----" * 8)
        top_movies_user = (
            movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
        )
        movie_df_rows = self.movies[self.movies["movieId"].isin(top_movies_user)]
        for row in movie_df_rows.itertuples():
            print(row.title, ":", row.genres)
        print("----" * 8)
        print("Top movie recommendations")
        print("----" * 8)
        recommended_movies = self.movies[self.movies["movieId"].isin(recommendations)]
        for row in recommended_movies.itertuples():
            print(row.title, ":", row.genres)


# Initialize the recommender system
recommend = personalisedSearcher()

# Define the root route
@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /recommend endpoint with user_id and query'})

# Define the recommendation route
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    if request.method == 'POST':
        data = request.json
        user_id = data.get('user_id')
        query = data.get('query')

        if user_id is not None and query is not None:
            recommendations, _ = recommend.personalised_search(user_id, query)
            recommended_movies = recommend.movies[recommend.movies["movieId"].isin(recommendations)]
            result = recommended_movies[['movieId', 'title', 'genres']].to_dict(orient='records')
            return jsonify(result)
        else:
            return jsonify({'error': 'Please provide user_id and query'}), 400

 
x = datetime.datetime.now()

@app.route("/getmovieinfo", methods=["POST"])
def prompt():
    print()
    data = request.json
    print(data)
    title = data.get("title", "")


    if not title:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = """Can you get a URL of an images for the films Vampires (1998) and Pandorum (2009)
                    """

        response = model.generate_content(prompt)

        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": e}), 500
 
 
# This is just for me to see if it works lol
@app.route('/data')
def dummy():
    print('hi')
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
    }


# curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d "{\"user_id\": 42, \"query\": \"Horror films with zombies\"}"


if __name__ == '__main__':
    app.run(debug=True)

import json
import re

