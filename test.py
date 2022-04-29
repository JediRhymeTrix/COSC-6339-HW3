# for data
import pandas as pd

# for deep learning
from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix


# constants

DATA_PATH = 'data/'
MODEL_PATH = 'models/'
MAX_LEN = 30

'''

Read prepared test dataset

'''

# read dataset
df_test = pd.read_csv(DATA_PATH + 'test.csv')

print(df_test.head())

# get X
X_test = df_test['text'].values

# get other numerical and categorical features
feat_test = df_test.drop(['text', 'y'], axis=1).values

# get target
y_test = df_test["y"].values

# preparing sequences from test data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_test)

X_test_indices = tokenizer.texts_to_sequences(X_test)

# padding sequences to uniform length
X_test_indices = pad_sequences(X_test_indices, maxlen=MAX_LEN)

'''

Load saved model (text-only)

'''

bst_model_path = MODEL_PATH + 'bid_lstm.h5'

# load model
model = models.load_model(bst_model_path)

model.summary()

'''

Evaluation

'''

# evaluate
scores = model.evaluate(X_test_indices, y_test)

# test
predictions = model.predict(X_test_indices)
predictions = list(map(lambda x: 1 if x > 0.5 else 0, predictions))

# confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print('\n-----------------------------------\n')
print('Model: text-only')
print('Accuracy: %.2f%%\n' % (scores[1]*100))
print('Confusion matrix: ')
print(conf_matrix)
print('\n-----------------------------------\n')

'''

Load saved model (text + additional features)

'''

bst_model_path = MODEL_PATH + 'bid_lstm_feats.h5'

# load model
model = models.load_model(bst_model_path)

model.summary()

'''

Evaluation

'''

# evaluate
scores = model.evaluate([X_test_indices, feat_test], y_test)

# test
predictions = model.predict([X_test_indices, feat_test])
predictions = list(map(lambda x: 1 if x > 0.5 else 0, predictions))

# confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print('\n-----------------------------------\n')
print('Model: text + additional features')
print('Accuracy: %.2f%%\n' % (scores[1]*100))
print('Confusion matrix: ')
print(conf_matrix)
print('\n-----------------------------------\n')

