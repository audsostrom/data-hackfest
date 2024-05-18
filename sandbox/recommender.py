import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
from annoy import AnnoyIndex


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


class personalisedSearcher:
    def __init__(self):
        self.movies = pd.read_csv("data/grouplens/ml-25m/movies.csv")
        self.ratings = pd.read_csv("data/grouplens/ml-25m/filtered_ratings.csv")
        self.embeddings = pd.read_csv("embeddings/data.csv", index_col=0)
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
        self.recommender.load_state_dict(torch.load('CF/CF.pth'))
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
        nearest_ids = self.index.get_nns_by_vector(query_embeddings[0].numpy(), 5)  # Find 5 nearest neighbors

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


recommend = personalisedSearcher()
recommend.print_recs(42, "Horror films with zombies")
