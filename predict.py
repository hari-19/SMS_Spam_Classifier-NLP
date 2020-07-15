import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = tf.keras.models.load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 189

text = input("Enter your sms : ")

tokenized_text = tokenizer.texts_to_sequences([text])[0]
tokenized_text = pad_sequences([tokenized_text], maxlen = max_sequence_len, padding='post')
tokenized_text = np.array(tokenized_text)
predicted = model.predict(tokenized_text, verbose=0)

print("\n\n\n")
if predicted > 0.5:
    print("Spam")
else:
    print("Not Spam")