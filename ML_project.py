# Import required libraries
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load train.csv into a Pandas dataframe
train_df = pd.read_csv('trainDataMLP.csv')

# Preprocess text data
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    # Tokenize text and remove stop words and punctuation
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)


train_df['processed_text'] = train_df['TITLE'].fillna('') + ' ' + train_df['DESCRIPTION'].fillna('') + ' ' + train_df['BULLET_POINTS'].fillna('')
train_df['processed_text'] = train_df['processed_text'].apply(preprocess_text)

# Combine preprocessed text and categorical data
vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
X_text = vectorizer.fit_transform(train_df['processed_text'])
X = np.concatenate((X_text.toarray(), train_df['PRODUCT_TYPE_ID'].values.reshape(-1, 1)), axis=1)
y = train_df['PRODUCT_LENGTH']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Set up early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=128, callbacks=[early_stop])

# Evaluate the model on the validation set
mse = model.evaluate(X_val, y_val)
mae = np.mean(np.abs(model.predict(X_val) - y_val))
rmse = np.sqrt(mse)

print("Mean squared error: {:.2f}".format(mse))
print("Mean absolute error: {:.2f}".format(mae))
print("Root mean squared error: {:.2f}".format(rmse))

# Load test.csv into a Pandas dataframe and preprocess the data
test_df = pd.read_csv('testDataMLP.csv')
test_df['processed_text'] = test_df['TITLE'].fillna('') + ' ' + test_df['DESCRIPTION'].fillna('') + ' ' + test_df['BULLET_POINTS'].fillna('')
test_df['processed_text'] = test_df['processed_text'].apply(preprocess_text)
X_test_text = vectorizer.transform(test_df['processed_text'])
X_test = np.concatenate((X_test_text.toarray(), test_df['PRODUCT_TYPE_ID'].values.reshape(-1, 1)), axis=1)

# Make predictions on the test set and generate a submission file
y_test_pred = mlp_reg.predict(X_test)
submission_df = pd.DataFrame( {'PRODUCT_ID': test_df['PRODUCT_ID'], 'PRODUCT_LENGTH': y_test_pred} )
submission_df.to_csv('submission.csv', index=False)
