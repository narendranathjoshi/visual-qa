from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM,GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys


f=open("questions_all.txt",'r')
total = f.readlines()
text = ""
test = ""
total = total#[:10000]
total_len = len(total)
train_len = int(total_len*0.7)
print (train_len)
test_len = len(total) - train_len
print (test_len)

for i in range(train_len):
    text += str(total[i]).lower()
for i in xrange(train_len+1,total_len):
    test += str(total[i]).lower()
print('corpus length:', len(text))

chars = sorted(list(set(text+test)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print (indices_char)
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 4
#step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print (len(sentences))


print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(GRU(32, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

perplexity = 0
for j in range(10):
    model.fit(X, y, batch_size=20, nb_epoch=1)


model_json = model.to_json()
with open("model_character_gru.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_character_gru.h5")
print("Saved model to disk")


N = 0
for i in range(len(test)-maxlen-1):
    sentence = test[i:i+maxlen]
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_char = test[i+maxlen]
    prob = preds[char_indices[next_char]]
    #print (prob,i)
    perplexity += np.log(prob)
    N += 1

print ("perplexity = ", perplexity)
perplexity /= -N
print ("-perplexity/N = ", perplexity)
print ("N = ", N)
perplexity = np.exp(perplexity)
print ("perplexity = ", perplexity)

exit(1)
