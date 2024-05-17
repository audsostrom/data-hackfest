from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hr-wesbeaver/themetagsv1", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("hr-wesbeaver/themetagsv1")

# example movie synopsis
synopsis = (
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
    "interstellar opens on earth in the distant future, as mother nature is waging war on humanity. famine is "
    "widespread, and all of mankind's resources are now dedicated to farming in a desperate fight for survival. a "
    "former nasa pilot and engineer named cooper (matthew mcconaughey) has become a homesteader in order to support "
    "his teenage son tom (timothee chalamet) and 10-year-old daughter murph (mackenzie foy). unfortunately, even that "
    "begins to look like a futile endeavor after a sandstorm ravages their close-knit community."
    "meanwhile, a seemingly supernatural mystery begins to unfold when the ""ghost"" that dwells in murph's room "
    "sends her a mysterious set of coordinates -- which lead the curious father/daughter duo to a clandestine "
    "underground base housing the remnants of nasa. there, renowned physicist professor brand (michael caine) "
    "has been working with a team of astronauts and scientists to find a new planet capable of sustaining human "
    "life. brand quickly persuades cooper to pilot a mission that will hopefully carry the seeds of human "
    "life to a habitable planet in another galaxy. cooper is soon bound for space with brand's daughter "
    "amelia (anne hathaway) and researchers doyle (wes bentley) and romilly (david gyasi)."
)

# tokenize the input text
inputs = tokenizer(synopsis, return_tensors="pt", truncation=True, padding="longest")

# generate themes
outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

# decode the generated themes
themes = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("|")

# print the themes
print(themes)
