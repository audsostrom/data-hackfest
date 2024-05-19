import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from annoy import AnnoyIndex
import time


# Function to get embedding for a query
def get_embedding(text, tokenizer, model, device):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = model_output.pooler_output
    embedding = torch.nn.functional.normalize(embedding)
    return embedding.cpu().numpy()


# Function to build Annoy index and run a query with specified number of trees
def build_and_query_annoy(query, n_trees, embeddings, data, tokenizer, model, device, k=10):
    # Get the query embedding
    query_embedding = get_embedding(query, tokenizer, model, device)[0]

    # Create and build the Annoy index
    embedding_dim = embeddings.shape[1]
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    for i, vector in enumerate(embeddings):
        annoy_index.add_item(i, vector)

    start_time = time.time()
    annoy_index.build(n_trees)
    build_time = time.time() - start_time

    print(f"Built Annoy index with {n_trees} trees in {build_time:.2f} seconds")

    search_k = 100 * k  # Adjust based on performance needs
    start_time = time.time()
    nearest_neighbors_indices = annoy_index.get_nns_by_vector(query_embedding, k, include_distances=True, search_k=search_k)
    query_time = time.time() - start_time

    # Initialize an empty list to store the information
    neighbor_info = []

    # Loop through the indices of the nearest neighbors
    for idx in nearest_neighbors_indices[0]:
        # Extract the title and genres for the current index
        movie_info = data.iloc[idx][['movieId', 'title', 'genres']]
        # Append the extracted information as a dictionary to the list
        neighbor_info.append(movie_info.to_dict())

    # Create a DataFrame from the collected information
    neighbors_df = pd.DataFrame(neighbor_info)

    # Print the DataFrame and query time
    print(f"Nearest neighbors using {n_trees} trees (query time: {query_time:.2f} seconds):")
    print(neighbors_df)
    print("\n")


# Function to test different numbers of trees
def test_annoy_trees(query, tree_values, embeddings, data, tokenizer, model, device, k=10):
    # Iterate over the range of tree values
    for n_trees in tree_values:
        print(f"Testing with {n_trees} trees...")
        build_and_query_annoy(query, n_trees, embeddings, data, tokenizer, model, device, k)


# Set options to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 200)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v2")
# model = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v2")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load movie data
data = pd.read_csv("data/grouplens/ml-25m/movies.csv")

# Load precomputed embeddings
embeddings_df = pd.read_csv("embeddings/labse_themes.csv", index_col=0)
embeddings = embeddings_df.values

# Example testing
query_text = "movie about aliens"
tree_values_to_try = [100, 200, 300, 400, 500]
# test_annoy_trees(query_text, tree_values_to_try, embeddings, data, tokenizer, model, device)

# Example usage
build_and_query_annoy(query_text, 200, embeddings, data, tokenizer, model, device)