from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Load the saved model
model = load_model('CNN_model.h5')

# Example texts to predict
sample_texts = ["first delivery - wheeler operating please see attached letter"]

with open('tokenizer.json', 'rb') as f:
    tokenize = pickle.load(f)
# Tokenize and pad the input texts
sample_sequences = tokenize.texts_to_sequences(sample_texts)
sample_padded = pad_sequences(sample_sequences, maxlen=200)

# Predict
predictions = model.predict(sample_padded)

# Convert probabilities to binary labels
print("Spam" if predictions[0] > 0.5 else "Ham")
