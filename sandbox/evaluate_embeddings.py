import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import re
import numpy as np
from annoy import AnnoyIndex

k = 10  # Number of nearest neighbors to retrieve
n_trees = 100


# Function to get embedding for a query
def get_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = model_output.pooler_output
    embedding = torch.nn.functional.normalize(embedding)
    return embedding.cpu().numpy()


# Set options to display all columns
pd.set_option('display.max_columns', None)
# Set options to display more rows, e.g., 100 rows
pd.set_option('display.max_rows', 100)
# If you also want to expand the width of each column to avoid truncation of data
pd.set_option('display.max_colwidth', None)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load movie data
data = pd.read_csv("data/grouplens/ml-25m/movies.csv")

# Load precomputed embeddings
embeddings_df = pd.read_csv("embeddings/title_genre.csv", index_col=0)
embeddings = embeddings_df.values

# Create Annoy index
embedding_dim = embeddings.shape[1]
annoy_index = AnnoyIndex(embedding_dim, 'angular')
for i, vector in enumerate(embeddings):
    annoy_index.add_item(i, vector)
annoy_index.build(n_trees)
print("done clustering!")
print()


# Test query
test = "horror films with zombies"
query_embedding = get_embedding(test)[0]

# Retrieve nearest neighbors using Annoy
search_k = n_trees * k
nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, k, include_distances=True, search_k=search_k)

# Initialize an empty list to store the information
neighbor_info = []

# Loop through the indices of the nearest neighbors
for idx in nearest_neighbors[0]:
    # Extract the title and genres for the current index
    info = data.iloc[idx][['movieId', 'title', 'genres']]
    # Append the extracted information as a dictionary to the list
    neighbor_info.append(info.to_dict())

# Create a DataFrame from the collected information
neighbors_df = pd.DataFrame(neighbor_info)

# Print the DataFrame
print(neighbors_df)
