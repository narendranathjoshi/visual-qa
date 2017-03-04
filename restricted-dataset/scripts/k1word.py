from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys
import string


f=open("questions.txt",'r')
lines = f.readlines()
lines = lines[:100000]
totalLen = len(lines)
trainLen = int(totalLen *0.7)
testLen = totalLen - trainLen
train=[]
test=[]
words=[]
i=0
maxlen=2

for line in lines:
   lastChar = line.strip()[-1]#"".join(l for l in line if l not in string.punctuation)
   line = "<s> "+line.strip()[:-1].lower() + " " + lastChar + " </s>"
   #import pdb;pdb.set_trace()
   w = line.split()
   words += w
   if i<trainLen:
       train.append(line)
   else:
       test.append(line)
   i+=1


words = list(set(words))
print (words)

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

sentences = []
next_word = []
for line in train:
    w = line.split()
    for i in range(0, len(w) - maxlen):
        sentences.append(w[i: i + maxlen])
        next_word.append(w[i + maxlen])
print('nb sequences:', len(sentences))
print (len(sentences))


print('Vectorization...')
print(len(sentences))
print(maxlen)
print(len(words))
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(next_word), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_word[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(32, input_shape=(maxlen, len(words))))
model.add(Dense(len(words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
for j in range(10):
    model.fit(X, y, batch_size=1024, nb_epoch=1)


model_json = model.to_json()
with open("model_word_lstm.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_word_lstm.h5")
print("Saved model to disk")

