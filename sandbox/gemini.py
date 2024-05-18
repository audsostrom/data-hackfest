from transformers import pipeline

generator = pipeline('text-generation', model='describeai/gemini')
input_prompt = ("Based on you liking the movies  "
                "It, Scream, The Conjuring, Amityville Horror, and Bride of Chucky; I think you would like: ")

response = generator(input_prompt, max_length=100, num_return_sequences=1)
print("Generated response: " + response[0]['generated_text'])
