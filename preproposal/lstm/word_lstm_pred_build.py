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


f=open("questions_ordered.txt",'r')
lines = f.readlines()
f=open("answers_ordered.txt", 'r')
answers = f.readlines()
lines = lines[:100000]
totalLen = len(lines)
trainLen = int(totalLen *0.7)
testLen = totalLen - trainLen
train=[]
test=[]
words=[]
i=0
maxlen=10

answerWords = []

for i in range (len(lines)):
   line = lines[i] #question
   answer = tuple(answers[i].strip().split()[:3])
   line = line.strip()[:-1].lower()
   #import pdb;pdb.set_trace()
   w = line.split()[:10]
   words += w
   answerWords.append(answer)
   if i<trainLen:
       train.append((line, answer))
   else:
       test.append((line, answer))
   i+=1


words = list(set(words))
answerWords = list(set(answerWords))
print (words)
print (answerWords)

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

ans_word_indices = dict((c, i) for i, c in enumerate(answerWords))
ans_indices_word = dict((i, c) for i, c in enumerate(answerWords))

#reading answer 

sentences = []
next_word = []
for (line, ans) in train:
    w = line.split()[:10]
    sentences.append(w)
    next_word.append(ans)
print('nb sequences:', len(sentences))
print (len(sentences))


print('Vectorization...')
print(len(sentences))
print(maxlen)
print(len(words))
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(next_word), len(answerWords)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, ans_word_indices[next_word[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(words))))
model.add(Dense(len(answerWords)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#for j in range(10):
model.fit(X, y, batch_size=1024, nb_epoch=10)


model_json = model.to_json()
with open("model_word_pred_lstm_128.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_word_pred_lstm_128.h5")
print("Saved model to disk")

