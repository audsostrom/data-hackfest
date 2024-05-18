from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hr-wesbeaver/themetagsv1", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("hr-wesbeaver/themetagsv1")

# Example movie synopsis
synopsis1 = (
    "Visionary filmmaker Christopher Nolan (Memento, The Dark Knight) writes and directs this psychological "
    "sci-fi action film about a thief who possesses the power to enter into the dreams of others. Dom Cobb "
    "(Leonardo DiCaprio) doesn't steal things, he steals ideas. By projecting himself deep into the "
    "subconscious of his targets, he can glean information that even the best computer hackers can't get to. "
    "In the world of corporate espionage, Cobb is the ultimate weapon. But even weapons have their weakness, "
    "and when Cobb loses everything, he's forced to embark on one final mission in a desperate quest for "
    "redemption. This time, Cobb won't be harvesting an idea, but sowing one. Should he and his team of "
    "specialists succeed, they will have discovered a new frontier in the art of psychic espionage. They've "
    "planned everything to perfection, and they have all the tools to get the job done. Their mission is "
    "complicated, however, by the sudden appearance of a malevolent foe that seems to know exactly what "
    "they're up to, and precisely how to stop them."
)

synopsis2 = (
    "Interstellar opens on earth in the distant future, as mother nature is waging war on humanity. Famine is "
    "widespread, and all of mankind's resources are now dedicated to farming in a desperate fight for survival. A "
    "former NASA pilot and engineer named Cooper (Matthew McConaughey) has become a homesteader in order to support "
    "his teenage son Tom (Timothee Chalamet) and 10-year-old daughter Murph (Mackenzie Foy). Unfortunately, even that "
    "begins to look like a futile endeavor after a sandstorm ravages their close-knit community. Meanwhile, a seemingly "
    "supernatural mystery begins to unfold when the 'ghost' that dwells in Murph's room sends her a mysterious set of "
    "coordinates -- which lead the curious father/daughter duo to a clandestine underground base housing the remnants of NASA. "
    "There, renowned physicist Professor Brand (Michael Caine) has been working with a team of astronauts and scientists to find "
    "a new planet capable of sustaining human life. Brand quickly persuades Cooper to pilot a mission that will hopefully carry "
    "the seeds of human life to a habitable planet in another galaxy. Cooper is soon bound for space with Brand's daughter Amelia "
    "(Anne Hathaway) and researchers Doyle (Wes Bentley) and Romilly (David Gyasi)."
)


# Function to get themes for a single synopsis
def get_themes(synopsis):
    #print(synopsis)
    if not pd.isna(synopsis):
        inputs = tokenizer(synopsis, return_tensors="pt", truncation=True, padding="longest")
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        themes = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("|")
        themes = [item.lower() for item in themes]
    else:
        themes = None
    #print(f"Themes: {themes}")
    #print("-" * 100)
    print(f"processed synopsis")
    return themes


# Function to process a chunk of data and save progress
def process_chunk(data, start_index, end_index, save_path):
    chunk = data.iloc[start_index:end_index].copy()  # Make an explicit copy of the slice
    chunk.loc[:, 'webeaver_allmovie_themes'] = chunk['allmovie_synopsis'].apply(get_themes)
    chunk.loc[:, 'webeaver_kaggle_themes'] = chunk['kaggle_overview'].apply(get_themes)
    # Save the processed chunk to a CSV file
    chunk.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)


data = pd.read_csv("clean_data.csv")

# Process data in chunks
chunk_size = 10
save_path = 'clean_data_themes.csv'

# Initialize the CSV file with the new columns
initial_columns = data.columns.tolist() + ['webeaver_allmovie_themes', 'webeaver_kaggle_themes']
pd.DataFrame(columns=initial_columns).to_csv(save_path, index=False)

for start_index in range(0, len(data), chunk_size):
    end_index = min(start_index + chunk_size, len(data))
    process_chunk(data, start_index, end_index, save_path)
