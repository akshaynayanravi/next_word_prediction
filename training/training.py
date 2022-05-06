import string
import numpy as np
from numpy import append
import pickle5 as pickle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer   
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam


file = open("/home/terralogic/akshaynayanravi/next_word_prediction/data/text_sample_data.txt")
raw_data = []
for i in file:
    raw_data.append(i)
# print(raw_data)

# CLEANING THE DATA...
data = ""
for i in raw_data:
    data = " ".join(raw_data)
data = data.replace("\n", "").replace("\r", "").replace("\ufeff", "")
# print(data[:100])
temp = []
for i in data.split():
    if i not in temp:
        temp.append(i)
data = " ".join(temp)
# print(data)


# translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
# new_data = data.translate(translator)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
sequence_data = tokenizer.texts_to_sequences([data])[0]
# print(sequence_data[:10])
vocab_size = len(tokenizer.word_index) + 1
# print(vocab_size)

sequences = []
for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)

# print(len(sequences))
sequences = np.array(sequences)
# print(sequences[:10])

x = []
y = []

for i in sequences:
    x.append(i[0])
    y.append(i[1])

x = np.array(x)
y = np.array(y)
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))
print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))

model.fit(x, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])
