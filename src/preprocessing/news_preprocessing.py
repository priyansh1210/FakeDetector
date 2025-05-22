import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('D:\\FakeDetector\\data\\news_data\\news.csv')

# Check data
print(df.shape)
print(df.head())

# Get the labels
labels = df.label

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)

# Initialize TfidfVectorizer with stop words
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Save preprocessed data
np.save('D:\\FakeDetector\\data\\news_data\\tfidf_train.npy', tfidf_train.toarray())
np.save('D:\\FakeDetector\\data\\news_data\\tfidf_test.npy', tfidf_test.toarray())
np.save('D:\\FakeDetector\\data\\news_data\\y_train.npy', y_train.values)
np.save('D:\\FakeDetector\\data\\news_data\\y_test.npy', y_test.values)
