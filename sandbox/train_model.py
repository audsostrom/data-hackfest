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


class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


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


def init_weights(m):
    if isinstance(m, nn.Embedding):
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.bias, 0)


def train():
    training_start = time.time()
    # initialize a W&B run, automatically using config from the sweep
    wandb.init()
    config = wandb.config
    run_name = f"lr_{config.learning_rate:.2e}-bs_{config.batch_size}-emb_{config.embedding_size}-ep_{config.epochs}"
    wandb.run.name = run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading and Preprocessing
    df = pd.read_csv("data/grouplens/ml-25m/filtered_ratings.csv")
    user_ids = df['userId'].unique().tolist()
    movie_ids = df['movieId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    df['user'] = df['userId'].map(user2user_encoded)
    df['movie'] = df['movieId'].map(movie2movie_encoded)
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    df['rating'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    msk = np.random.rand(len(df)) < 0.9
    train_df = df[msk]
    val_df = df[~msk]

    # Dataset and DataLoader
    train_dataset = MovieDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = MovieDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model setup
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    model = RecommenderNet(num_users, num_movies, config.embedding_size).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(config.epochs):
        epoch_start = time.time()
        print(f"Starting Epoch {epoch + 1}")
        model.train()
        total_loss = 0
        for i, (user, movie, rating) in enumerate(train_loader):
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, movie)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 500 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")
                wandb.log({'batch_loss': loss.item(), "epoch": epoch, "batch_num": i})
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch, "training_loss": avg_train_loss})

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (user, movie, rating) in enumerate(val_loader):
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                output = model(user, movie)
                val_loss = criterion(output, rating)
                total_val_loss += val_loss.item()
                pass
        avg_val_loss = total_val_loss / (i + 1)
        epoch_duration = (time.time() - epoch_start) / 60
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}. Epoch Duration: {epoch_duration:.2f} minutes")
        wandb.log({"epoch": epoch, "validation_loss": avg_val_loss, "epoch_duration": epoch_duration})

    training_duration = (time.time() - training_start) / 60
    print(f"Total training time (minutes): {training_duration:.2f} minutes")
    wandb.log({'Total Training Time (minutes)': training_duration})
    wandb.finish()
    print("-"*250)


def sweep():
    timestamp = datetime.now().strftime("%m.%d_%H.%M")
    sweep_config = {
        "method": "random",
        "metric": {"name": "batch_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 0.00025, "max": 0.00050},
            "epochs": {"distribution": "q_uniform", "min": 5, "max": 10, "q": 1},
            "batch_size": {"values": [512, 1024, 2048, 4096, 8192]},
            "embedding_size": {"values": [64, 128, 256, 512, 1024]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=f"2024-DataHackfest")
    wandb.agent(sweep_id, function=train, count=5)


def run_single_experiment():
    # Define hyperparameters directly within the function
    config = {
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 4096,
        "embedding_size": 128
    }

    # Start a wandb run with the given configuration
    wandb.init(project="single_run_experiments", config=config)
    run_name = f"lr_{config['learning_rate']}-bs_{config['batch_size']}-emb_{config['embedding_size']}-ep_{config['epochs']}"
    wandb.run.name = run_name

    training_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading and Preprocessing
    df = pd.read_csv("data/grouplens/ml-25m/filtered_ratings.csv")
    user_ids = df['userId'].unique().tolist()
    movie_ids = df['movieId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    df['user'] = df['userId'].map(user2user_encoded)
    df['movie'] = df['movieId'].map(movie2movie_encoded)
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    df['rating'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    msk = np.random.rand(len(df)) < 0.9
    train_df = df[msk]
    val_df = df[~msk]

    train_dataset = MovieDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = MovieDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    model = RecommenderNet(num_users, num_movies, config['embedding_size']).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        print(f"Starting Epoch {epoch + 1}")
        model.train()
        total_loss = 0
        for i, (user, movie, rating) in enumerate(train_loader):
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, movie)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")
                wandb.log({'batch_loss': loss.item(), "epoch": epoch, "batch_num": i})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch, "training_loss": avg_train_loss})

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (user, movie, rating) in enumerate(val_loader):
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                output = model(user, movie)
                val_loss = criterion(output, rating)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_duration = (time.time() - epoch_start) / 60
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}. Epoch Duration: {epoch_duration:.2f} minutes")
        wandb.log({"epoch": epoch, "validation_loss": avg_val_loss, "epoch_duration": epoch_duration})

    training_duration = (time.time() - training_start) / 60
    print(f"Total training time (minutes): {training_duration:.2f} minutes")
    wandb.log({'Total Training Time (minutes)': training_duration})
    wandb.finish()
    torch.save(model.state_dict(), 'CF/CF.pth')
    print("Model saved as CF.pth")


if __name__ == "__main__":
    # run_single_experiment()
    sweep()
