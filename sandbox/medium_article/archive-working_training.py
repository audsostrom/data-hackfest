import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import wandb
from datetime import datetime

# Setup Weights & Biases
timestamp = datetime.now().strftime("%m.%d_%H.%M")
os.environ["WANDB_PROJECT"] = "2024-DataHackfest"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "false"
wandb.init()
config = wandb.config
config.embedding_size = 128
config.epochs = 2   # was 5
config.batch_size = 4096
config.learning_rate = 0.001
wandb.run.name = f"lr_{config.learning_rate}-epochs_{config.epochs}-batch_{config.batch_size}-embedd_{config.embedding_size}--{timestamp}"
wandb.run.save()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Read and preprocess data
df = pd.read_csv("../../data/grouplens/ml-25m/ratings.csv")
print("Data loaded and preprocessing started...")
user_ids = df['userId'].unique().tolist()
movie_ids = df['movieId'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
df['user'] = df['userId'].map(user2user_encoded)
df['movie'] = df['movieId'].map(movie2movie_encoded)

min_rating = df['rating'].min()
max_rating = df['rating'].max()
df['rating'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))

# Split data
np.random.seed(42)
msk = np.random.rand(len(df)) < 0.9
train_df = df[msk]
val_df = df[~msk]
print("Data split into training and validation sets.")


class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


train_dataset = MovieDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

val_dataset = MovieDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


class RecommenderNet(nn.Module):
    # def __init__(self, num_users, num_movies, embedding_size):
    #     super(RecommenderNet, self).__init__()
    #     self.user_embedding = nn.Embedding(num_users, embedding_size)
    #     self.user_bias = nn.Embedding(num_users, 1)
    #     self.movie_embedding = nn.Embedding(num_movies, embedding_size)
    #     self.movie_bias = nn.Embedding(num_movies, 1)
    #     self.user_bn = nn.BatchNorm1d(embedding_size)  # Added
    #     self.movie_bn = nn.BatchNorm1d(embedding_size)  # Added
    #
    # def forward(self, user_input, movie_input):
    #     user_vec = self.user_bn(self.user_embedding(user_input))  # Added
    #     user_bias = self.user_bias(user_input)
    #     movie_vec = self.movie_bn(self.movie_embedding(movie_input))  # Added
    #     movie_bias = self.movie_bias(movie_input)
    #     dot = (user_vec * movie_vec).sum(1)
    #     output = torch.sigmoid(dot + user_bias.squeeze() + movie_bias.squeeze())
    #     return torch.clamp(output, min=1e-7, max=1 - 1e-7)  # Added clamping

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


num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)
model = RecommenderNet(num_users, num_movies, config.embedding_size).to(device)

# Initialize weights
# def init_weights(m):
#     if type(m) == nn.Embedding:
#         torch.nn.init.uniform_(m.weight, -0.01, 0.01)
#     elif type(m) == nn.Linear:
#         torch.nn.init.constant_(m.bias, 0)
def init_weights(m):
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.bias, 0)


model.apply(init_weights)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

training_start = time.time()
for epoch in range(config.epochs):
    print(f"Starting Epoch {epoch + 1}")
    model.train()
    running_loss = 0.0
    for i, (user, movie, rating) in enumerate(train_loader):
        #print("Sample ratings before entering the model:", rating[:5])
        user, movie, rating = user.to(device), movie.to(device), rating.to(device)
        optimizer.zero_grad()
        output = model(user, movie)
        loss = criterion(output, rating)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")
            #print("Sample outputs:", output[:5].detach().cpu().numpy())
            #print("Corresponding ratings:", rating[:5].cpu().numpy())
            #print("-"*100)
            wandb.log({'Batch Loss': loss.item()})

    # Validation logic here...
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for user, movie, rating in val_loader:
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            output = model(user, movie)
            loss = criterion(output, rating)
            val_loss += loss.item()
            pass

    average_loss = running_loss / len(train_loader)
    average_val_loss = val_loss / len(val_loader)  # Calculate average validation loss
    print(f"Epoch {epoch + 1} Training Average Loss: {average_loss:.4f}, Validation Average Loss: {average_val_loss:.4f}")
    wandb.log({'Epoch': epoch+1, 'Training Average Loss': average_loss, 'Validation Average Loss': average_val_loss})
training_duration = (time.time() - training_start) / 60
print(f"Total training time (minutes): {training_duration:.2f} minutes")
wandb.log({'Total Training Time (minutes)': training_duration})

# Save the model
torch.save(model.state_dict(), '../CF/CF.pth')
