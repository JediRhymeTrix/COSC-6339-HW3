# for data
import pandas as pd
import numpy as np

# for deep learning
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Permute, multiply, Activation, Input, concatenate
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

# get other numerical and categorical features
feat_train = df_train.drop(['text', 'y'], axis=1).values

# get target
y_train = df_train["y"].values

'''

Load prepared word embeddings

'''

emb_matrix = np.genfromtxt(DATA_PATH + 'word_emb.csv', delimiter=',')

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
def bid_lstm(input_shape, additional_input_shape=None):
    # input
    X_indices = Input(input_shape, name="text_input")
    inputs = X_indices

    # additional input layer of other features besides text
    if additional_input_shape:
        feat_input = Input(additional_input_shape, name='feat_input')
        inputs = [X_indices, feat_input]
        feat_input = Activation('tanh')(feat_input)

    # embedding
    embeddings = embedding_layer(X_indices)

    # apply attention
    X = attention_layer(embeddings, neurons=MAX_LEN)

    # 2 layers of bidirectional lstm
    X = Bidirectional(LSTM(
        units=MAX_LEN, dropout=0.2, return_sequences=True))(X)
    X = Bidirectional(LSTM(units=MAX_LEN, dropout=0.2))(X)

    # concatenating additional features before prediction layer
    if 'feat_input' in locals():
        X = concatenate([X, feat_input])

    # final dense layers
    X = Dense(64, activation='relu')(X)
    y_out = Dense(1, activation='sigmoid')(X)

    # compile
    model = Model(inputs=inputs, outputs=y_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# tokenizing  sequences from training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_indices = tokenizer.texts_to_sequences(X_train)

# padding sequences to uniform length
X_train_indices = pad_sequences(X_train_indices, maxlen=MAX_LEN)

'''

1. Text-only model

'''

# model for text features only
model = bid_lstm((MAX_LEN,))

print('Text-only model')
model.summary()

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
print('\n-----------------------------------\n')

'''

2. Text + additional features model

'''

# model for text + additional features
model = bid_lstm((MAX_LEN,), (feat_train.shape[1],))

print('Text + additional features model')
model.summary()

# early stopping and model checkpoints

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = MODEL_PATH + 'bid_lstm_feats.h5'

model_checkpoint = ModelCheckpoint(
    bst_model_path, save_best_only=True, save_weights_only=False)

# training

# text + additional features model
hist_text_feat = model.fit([X_train_indices, feat_train], y_train, validation_split=0.2,
                           batch_size=2048, epochs=50, callbacks=[early_stopping, model_checkpoint])

bst_val_score = min(hist.history['val_loss'])

print('\n-----------------------------------\n')
print('best val score: {}'.format(bst_val_score))
print('model saved to {}'.format(bst_model_path))
print('\n-----------------------------------\n')

# Note: The models will be saved to disk after training. Run test.py to evaluate the saved models.
