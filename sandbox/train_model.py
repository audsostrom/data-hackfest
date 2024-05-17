import sys

import wandb
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
import os
import numpy as np
from transformers import SchedulerType
from datetime import datetime
from sklearn.metrics import f1_score

# hyperparameters
hp_learningrate = 4e-5
hp_batch_size = 16
hp_epochs = 10
hp_weightdecay = 0.01
hp_warmupsteps = 150
hp_learningratescheduler = SchedulerType.COSINE
# LINEAR
# COSINE
# COSINE_WITH_RESTARTS
# POLYNOMIAL
# CONSTANT
# CONSTANT_WITH_WARMUP

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="2024-DataHackfest"
# save trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

# download nltk data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ensure output directory exists
output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# function to compute metrics like f1 score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to logits and threshold at 0.5
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    # Compute metrics like F1 score which is better for multilabel
    f1_micro = f1_score(labels, predictions, average='micro')
    return {"f1_micro": f1_micro}


# callback to log metrics with wandb
class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Here you can log any metrics to wandb
        wandb.log(metrics)


# preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# adjust pandas display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks
pd.set_option('display.max_colwidth', 100)  # Show full content of each column

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# load data
print("Loading CSV...")
raw_data = pd.read_csv("test_clean_data.csv")
# raw_data = pd.read_csv("clean_data.csv")
print("CSV Loaded")

# select relevant columns and ensure no NaN values
data = raw_data[['kaggle_id', 'kaggle_title', 'allmovie_synopsis', 'allmovie_themes']].copy()
data['allmovie_themes'] = data['allmovie_themes'].fillna('')
data['allmovie_synopsis'] = data['allmovie_synopsis'].fillna('')

# clean data
print(f"Initial DataFrame size: {data.shape}")
data = data[(data['allmovie_themes'] != '') & (data['allmovie_synopsis'] != '')]
print(f"Size after cleaning: {data.shape}")

# preprocess and split themes
print("Preprocessing inputs...")
data['allmovie_synopsis_processed'] = data['allmovie_synopsis'].apply(preprocess_text)
print("Inputs preprocessed")
data['allmovie_themes'] = data['allmovie_themes'].str.lower().str.split(', ')
print("Themes normalized and split")

# display data
print("Data head:")
print(data.head())
print()

# one-hot encode themes and save encoder classes
mlb = MultiLabelBinarizer()
data['allmovie_themes_encoded'] = mlb.fit_transform(data['allmovie_themes']).tolist()
np.save(os.path.join(output_dir, 'mlb_classes.npy'), mlb.classes_)

# display encoded themes
print("One-hot encoded themes:")
print(data[['allmovie_themes', 'allmovie_themes_encoded']].head())
print()

# convert the encoded labels to DataFrame for easier handling
labels_df = pd.DataFrame(data['allmovie_themes_encoded'].tolist())
# prepare training and validation data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['allmovie_synopsis_processed'],
    labels_df,
    test_size=0.2,
    random_state=42
)
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# tokenize data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())

# check cuda availability
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")


# define dataset class
class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


# initialize datasets
train_dataset = MovieDataset(train_encodings, train_labels)
val_dataset = MovieDataset(val_encodings, val_labels)

# display dataset sizes
print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

# initialize model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(mlb.classes_))

# define training arguments
timestamp = datetime.now().strftime("%m_%d_%H_%M")
sched_string = str(hp_learningratescheduler).replace("SchedulerType.", "")
training_args = TrainingArguments(
    output_dir=output_dir,
    report_to="wandb",
    run_name=f"run_{timestamp}-lr_{hp_learningrate}-sched_{sched_string}-epoch_{hp_epochs}-batch_{hp_batch_size}"
             f"-wustep_{hp_warmupsteps}-decay_{hp_weightdecay}",
    num_train_epochs=hp_epochs,
    per_device_train_batch_size=hp_batch_size,
    per_device_eval_batch_size=hp_batch_size,
    warmup_steps=hp_warmupsteps,
    weight_decay=hp_weightdecay,
    learning_rate=hp_learningrate,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    lr_scheduler_type=hp_learningratescheduler
)

# initialize and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback()]
)

# start training
print("Starting training...")
trainer.train()
print("Training completed")

# save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# verify saved files
files = os.listdir(output_dir)
print("Files in output directory:", files)

# finish wandb logging
wandb.finish()
