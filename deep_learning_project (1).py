
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import sys

# Replace with the actual path to your library directory in Google Drive
library_path = '/content/drive/MyDrive/1'

if library_path not in sys.path:
    sys.path.append(library_path)

# Create folders
!mkdir -p data
!mkdir -p results
!mkdir -p embeddings

import pandas as pd
import matplotlib.pyplot as plt

# Load Yelp review JSON from mounted Drive or upload manually
json_file_path = '/content/drive/MyDrive/1/data/yelp_academic_dataset_review.json'  # Adjust path if using Drive

# Read data in chunks
df_list = []
chunk_size = 10000
desired_records = 20000
records_read = 0

chunks = pd.read_json(json_file_path, lines=True, chunksize=chunk_size)

for chunk in chunks:
    df_list.append(chunk)
    records_read += chunk.shape[0]
    if records_read >= desired_records:
        break

df = pd.concat(df_list, ignore_index=True)

# Select and rename columns
df = df[['business_id', 'stars', 'text']].rename(columns={'stars': 'labels', 'text': 'review'})
df = df.dropna(subset=['review'])

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df['labels'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
         edgecolor='black', align='mid', rwidth=0.8)
plt.title('Distribution of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig('results/histogram.png')
plt.show()

# Balance the dataset
min_class_count = 1000
balanced_df = df.groupby('labels', group_keys=False).apply(
    lambda x: x.sample(n=min_class_count, random_state=42)
).reset_index(drop=True)

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to pickle
balanced_df.to_pickle('/content/drive/MyDrive/1/data/yelp_processed.pkl')

# Preview
print(balanced_df.head())
print("Processed DataFrame rows:", balanced_df.shape[0])

import pandas as pd
import spacy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from encoders.BERT import BertEmbedder
from encoders.T5 import T5Embedder
from encoders.word2vec import Word2VecEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_spacy_model():
    import spacy.cli
    spacy.cli.download("en_core_web_md")

def process_model_chunked(model_name, embedder_class, df, chunk_size=100):
    try:
        embedder = embedder_class()
        all_embeddings = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            logger.info(f"{model_name}: Processing chunk {i}–{i+chunk_size}")

            chunk[f'{model_name.lower()}_embeddings'] = chunk['review'].apply(embedder.get_embeddings)
            chunk = chunk[chunk[f'{model_name.lower()}_embeddings'].notnull()]  # Drop failed rows

            all_embeddings.append(chunk)

        combined = pd.concat(all_embeddings, ignore_index=True)
        combined.to_pickle(f'embeddings/yelp_{model_name}_embeddings.pkl')
        logger.info(f"✅ Generated {model_name} embeddings")
    except Exception as e:
        logger.error(f"❌ Error processing {model_name}: {e}")

def main():
    download_spacy_model()
    df = pd.read_pickle('/content/drive/MyDrive/1/data/yelp_processed.pkl')

    embedders = [
        ('BERT', BertEmbedder),
        ('T5', T5Embedder),
        ('Word2Vec', Word2VecEmbedder),
    ]

    # Parallel using threads (safer for Colab)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_model_chunked, name, embedder, df.copy())
            for name, embedder in embedders
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from decoders.svm import SVMClassifier
from decoders.logistic_regression import LogisticRegressionClassifier
from decoders.CNN import CNNClassifier
from decoders.Gradient_Boosting import GradientBoostingClassifierWrapper
from decoders.MLP import MLPClassifier
from decoders.Random_Forest import RandomForestClassifierWrapper
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import inspect

def train_and_save_classifier(classifier, X, y, file_suffix):
    # Check if the classifier requires input_dim parameter
    if 'input_dim' in inspect.signature(classifier.__init__).parameters:
        input_dim = X.shape[1]  # Assuming X is a 2D array where each row corresponds to a sample
        model = classifier(input_dim=input_dim)  # Provide the input_dim argument
    else:
        model = classifier()

    # Train the classifier
    model.train(X, y)

    # Save the results
    model.save_results(f'results/yelp_{file_suffix}.pkl')

def load_and_process_embeddings(file_path, column_name):
    # Read embeddings from a pickle file
    df = pd.read_pickle(file_path)

    # Convert the 'column_name' to NumPy arrays
    df[column_name] = df[column_name].apply(lambda x: np.array(x))
    X = np.array(df[column_name].tolist())
    y = df['labels'].values

    return X, y

def main():
    # List of embedding types and corresponding column names
    embeddings_list = [
        ('Word2Vec', 'word2vec_embeddings'),
        ('BERT', 'bert_embeddings'),
        ('T5', 't5_embeddings')
    ]

    # List of classifiers and their suffixes
    classifiers = [
        (LogisticRegressionClassifier, 'logistic_regression'),
        (SVMClassifier, 'svm'),
        (MLPClassifier, 'mlp'),
        (CNNClassifier, 'cnn'),
        (GradientBoostingClassifierWrapper, 'gradient_boosting'),
        (RandomForestClassifierWrapper, 'random_forest')
    ]

    # Loop through each embedding type and classifier
    for embedding_type, column_name in embeddings_list:
        file_path = f'/content/embeddings/yelp_{embedding_type}_embeddings.pkl'
        X, y = load_and_process_embeddings(file_path, column_name)

        for classifier, suffix in classifiers:
            # Train and save the classifier
            train_and_save_classifier(classifier, X, y, f'{embedding_type}_{suffix}')

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import pickle

# Sample data for demonstration purposes (replace this with your actual data)
file_format = 'yelp_{}_{}.pkl'
encoders = ['Word2Vec', 'BERT', 'T5']
decoders = ['logistic_regression', 'svm', 'mlp', 'cnn', 'gradient_boosting', 'random_forest']

# Create DataFrames to store evaluation metrics
accuracy_data = pd.DataFrame(index=encoders, columns=decoders)
f1_data = pd.DataFrame(index=encoders, columns=decoders)
precision_data = pd.DataFrame(index=encoders, columns=decoders)
recall_data = pd.DataFrame(index=encoders, columns=decoders)
specificity_data = pd.DataFrame(index=encoders, columns=decoders)

for encoder in encoders:
    for decoder in decoders:
        file_path = Path('results') / file_format.format(encoder, decoder)

        try:
            with open(file_path, 'rb') as file:
                results_data = pickle.load(file)

            # Extract evaluation metrics from the results_data dictionary
            accuracy_value = results_data.get('accuracy', None)
            f1_value = results_data.get('weighted avg', {}).get('f1-score', None)
            precision_value = results_data.get('weighted avg', {}).get('precision', None)
            recall_value = results_data.get('weighted avg', {}).get('recall', None)



            # Update the DataFrames
            accuracy_data.loc[encoder, decoder] = accuracy_value
            f1_data.loc[encoder, decoder] = f1_value
            precision_data.loc[encoder, decoder] = precision_value
            recall_data.loc[encoder, decoder] = recall_value

        except FileNotFoundError:
            print(f"File not found: {file_path}")

# Display and save the tables
print("\nAccuracy Table:\n")
print(accuracy_data)
accuracy_csv_filename = 'results/accuracy_table.csv'
accuracy_data.to_csv(accuracy_csv_filename)
print(f"Accuracy table saved to {accuracy_csv_filename}")

print("\nF1 Table:\n")
print(f1_data)
f1_csv_filename = 'results/f1_table.csv'
f1_data.to_csv(f1_csv_filename)
print(f"F1 table saved to {f1_csv_filename}")

print("\nPrecision Table:\n")
print(precision_data)
precision_csv_filename = 'results/precision_table.csv'
precision_data.to_csv(precision_csv_filename)
print(f"Precision table saved to {precision_csv_filename}")

print("\nRecall Table:\n")
print(recall_data)
recall_csv_filename = 'results/recall_table.csv'
recall_data.to_csv(recall_csv_filename)
print(f"Recall table saved to {recall_csv_filename}")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import csv

class SVMClassifier:
    def __init__(self, input_dim=None):
        self.model = SVC()
        self.results = {'classification_report': None}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Record overall metrics
        y_test_pred = self.model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        self.results['classification_report'] = test_report

    def predict(self, input_embedding):
        # Make a prediction using the trained model
        return self.model.predict([input_embedding])

    def save_results(self, filename):
        # Save the overall metrics as a CSV file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            for class_label, metrics in self.results['classification_report'].items():
                if class_label.isnumeric():
                    writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

        print(f"Results saved to {filename}")

def train_and_save_classifier(classifier, X, y, file_suffix):
    model = classifier()
    model.train(X, y)

    # Save the results
    filename = f'results/yelp_full_{file_suffix}.csv'
    model.save_results(filename)

    # Display the classification report
    print("Classification Report:")
    print(model.results['classification_report'])

    return model  # Return the trained model

def load_and_process_embeddings(file_path, column_name):
    df = pd.read_pickle(file_path)
    df[column_name] = df[column_name].apply(np.array)
    X = np.array(df[column_name].tolist())
    y = df['labels'].values

    return X, y

def main():
    # load the model and get the metrics

    embeddings_list = [('BERT', 'bert_embeddings')]

    classifiers = [
        (SVMClassifier, 'svm'),
    ]

    trained_models = []

    for embedding_type, column_name in embeddings_list:
        file_path = f'embeddings/yelp_{embedding_type}_embeddings.pkl'
        X, y = load_and_process_embeddings(file_path, column_name)

        for classifier, suffix in classifiers:
            trained_model = train_and_save_classifier(classifier, X, y, f'{embedding_type}_{suffix}')
            trained_models.append((trained_model, embedding_type))

    print("How the model will be used to make Predictions")

    # review recommendation for restaurant with id 'M0c99tzIJPIbrY_RAO7KSQ'

    file_path = 'embeddings/yelp_BERT_embeddings.pkl'

    # Read the pickle file into a DataFrame
    df = pd.read_pickle(file_path)

    # Specify the business ID you want to filter
    target_business_id = 'M0c99tzIJPIbrY_RAO7KSQ'

    # Create a new DataFrame with only rows that match the specified business ID
    filtered_df = df[df['business_id'] == target_business_id]

    # Display the new DataFrame
    print(filtered_df.head())

    # Calculate the average of the 'bart_embeddings' column
    average_embedding = np.mean(np.stack(filtered_df['bert_embeddings']), axis=0)

    print("Average Embedding:", average_embedding)

    # Make predictions using the trained models
    for trained_model, embedding_type in trained_models:
        prediction = trained_model.predict(average_embedding)
        print(f"Prediction using {embedding_type} model:", prediction)
        if prediction[0] == 1:
            print("Really don't go to this restaurant!")
        elif prediction[0] == 2:
            print("I would advise against it.")
        elif prediction[0] == 3:
            print("It's average.")
        elif prediction[0] == 4:
            print("I would advise for it.")
        elif prediction[0] == 5:
            print("Go to this restaurant!")

if __name__ == "__main__":
    main()
