# Transformer-Based-Contextual-Model-for-Sentiment-Analysis
This project performs sentiment analysis on Yelp review data using a deep learning pipeline that includes preprocessing, embedding generation, and classification using a variety of ML models.
i. Processes 20,000 Yelp reviews and balances the dataset for equal star ratings.
ii. Extracts features using multiple embedding techniques:
BERT
T5
Word2Vec

iii. Trains classifiers on embeddings using:
Logistic Regression
Support Vector Machine (SVM)
Multilayer Perceptron (MLP)
Random Forest
Gradient Boosting
Convolutional Neural Network (CNN)
Parallel embedding generation for efficient performance in Google Colab.

iv.Technologies & Tools
Python, Pandas, NumPy, Matplotlib
Scikit-learn
Spacy (for preprocessing)
Pretrained transformer models (BERT, T5)
google Colab for execution
Custom modular encoder and decoder architecture

V. Structure:

├── data/                # Yelp review dataset
├── embeddings/          # Stored embeddings for each model
├── results/             # Classifier performance results
├── encoders/            # Embedding model wrappers (BERT, T5, Word2Vec)
├── decoders/            # Classifier wrappers (SVM, MLP, CNN, etc.)
└── deep_learning_project.py

🔍 Visualization
Evaluation and Results
We evaluated a combination of three embedding models and six classification algorithms to determine the most effective setup for Yelp review sentiment analysis. Each model was assessed using multiple performance metrics: Accuracy, F1 Score, Precision, and Recall.

🔍 Key Findings
From the performance tables:
BERT + SVM emerged as the most consistent and best-performing combination, with:
Accuracy: 0.573
F1 Score: 0.568
Precision: 0.572
Recall: 0.573
T5 + MLP also showed promising results, especially in F1 Score (0.553) and Precision (0.565), but overall did not outperform the BERT-based setup.
Word2Vec embeddings underperformed compared to the transformer-based methods (BERT and T5), showing the power of contextual embeddings for this task.
