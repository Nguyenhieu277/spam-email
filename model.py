import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from SRC.CNN_model import CNN
import pickle
df = pd.read_csv("dataset/spam_ham_dataset.csv")

X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenize = Tokenizer(num_words = 10000)
tokenize.fit_on_texts(X_train)
with open('tokenizer.json', 'wb') as f:
    pickle.dump(tokenize, f)    
X_train_seq = tokenize.texts_to_sequences(X_train)
X_test_seq = tokenize.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen = 200)
X_test_pad = pad_sequences(X_test_seq, maxlen = 200)

model = CNN(10000, 128, 200)

model.compile()
model.train_model(X_train_pad, y_train, X_test_pad, y_test, 5)

model.save('CNN_model.h5')