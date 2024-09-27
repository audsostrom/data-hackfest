# Cinemap
For MLH Data Hackfest 2024

[DevPost Link](https://devpost.com/software/cinemap)

## Inspiration
One of team members, Audrey, has serious indecision issues. She spends more hours scrolling through Netflix than actual movie-watching, annoying her friends on Friday movie nights. This app for all Audreys in the world. Now these Audreys can go do more productive things with that extra time.

## What it does
Cinemap leverages advanced AI to understand your unique tastes and preferences. It curates a selection of movies tailored just for you, saving you hours of indecision. Simply based on your watch history and recent movie reviews, the app will quickly predict some movies for you. You can also prompt our prediction system with what you're feeling in particular through a quick quiz for even further individualized results.

## How we built it
We used this [online dataset](https://grouplens.org/datasets/movielens/). With over 62,000 movies and 25M movie ratings made by 162,000 users, we had a lot of up-to-date data to pull from. We use movies from this dataset as the basis for our suggestions, and we also trained it on these movies and ratings. Our transformer model of choice was [LABSE](https://huggingface.co/sentence-transformers/LaBSE). Using this model, approximate nearest neighbors, neural collaborative filtering – we simulated user to movie interactions based on other user's ratings to see how similar user's tastes were.

From there, we developed a NextJS-powered platform for people to use. We're able to take the user's data and interact with the model via Flask.


