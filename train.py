# for data
import json

import pandas as pd
import numpy as np

# for deep learning
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Permute, multiply, Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# constants

DATA_PATH = 'data/'
MODEL_PATH = 'models/'
MAX_LEN = 30

'''

Read prepared train dataset

'''

# read dataset
df_train = pd.read_csv(DATA_PATH + 'train.csv')

print(df_train.head())

# get X
X_train = df_train['text'].values

# get target
y_train = df_train["y"].values

'''

Load prepared word embeddings

'''

emb = json.load(open(DATA_PATH + 'words_emb.json'))

emb_matrix = np.array(list(emb.values()))
words = list(emb.keys())

embedding_layer = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[
                            emb_matrix], input_length=MAX_LEN, trainable=False)

'''

Define Bi-directional LSTM model

'''


# code attention layer
def attention_layer(inputs, neurons):
    X = Permute((2, 1))(inputs)
    X = Dense(neurons, activation="softmax")(X)
    X = Permute((2, 1), name="attention")(X)
    X = multiply([inputs, X])

    return X


# model
def bid_lstm(input_shape):
    # input
    X_indices = Input(input_shape)

    # embedding
    embeddings = embedding_layer(X_indices)

    # apply attention
    X = attention_layer(embeddings, neurons=MAX_LEN)

    # 2 layers of bidirectional lstm
    X = Bidirectional(LSTM(
        units=MAX_LEN, dropout=0.2, return_sequences=True))(X)
    X = Bidirectional(LSTM(units=MAX_LEN, dropout=0.2))(X)

    # final dense layers
    X = Dense(64, activation='relu')(X)
    y_out = Dense(1, activation='sigmoid')(X)

    # compile
    model = Model(inputs=X_indices, outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# build model
model = bid_lstm((MAX_LEN,))

model.summary()

'''

Training

'''

# tokenizing  sequences from training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_indices = tokenizer.texts_to_sequences(X_train)

# padding sequences to uniform length
X_train_indices = pad_sequences(X_train_indices, maxlen=MAX_LEN)

# early stopping and model checkpoints

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = MODEL_PATH + 'bid_lstm.h5'

model_checkpoint = ModelCheckpoint(
    bst_model_path, save_best_only=True, save_weights_only=False)

# training

hist = model.fit(X_train_indices, y_train, validation_split=0.2,
                 batch_size=2048, epochs=50, callbacks=[early_stopping, model_checkpoint])

bst_val_score = min(hist.history['val_loss'])

print('\n-----------------------------------\n')
print('best val score: {}'.format(bst_val_score))
print('model saved to {}'.format(bst_model_path))
print('run test.py to evaluate')

# Note: The model will be saved to disk after training. Run test.py to evaluate the saved model.
