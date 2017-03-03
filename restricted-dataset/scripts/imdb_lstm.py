
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 2  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

vectorSize = -99
for x_train in X_train:
    vectorSize = max(vectorSize, max(x_train))
print (vectorSize)

# for training
bigrams = []
next_word = []
for x_train in X_train:
    for i in range(0, len(x_train) - maxlen):
        bigrams.append(x_train[i: i + maxlen])
        next_word.append(x_train[i + maxlen])

print (len(bigrams))
print (vectorSize)
exit(1)

X = np.zeros((len(bigrams), maxlen, vectorSize), dtype=np.bool)
y = np.zeros((len(next_word), vectorSize), dtype=np.bool)
for i, bigram in enumerate(bigrams):
    for t, word in enumerate(bigram):
        X[i, t, word] = 1
    y[i, next_word[i]] = 1


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for j in range(1):
    model.fit(X, y, batch_size=1024, nb_epoch=1)


bigrams = []
next_word = []
for x_test in X_test:
    for i in range(0, len(x_test) - maxlen):
        bigrams.append(x_test[i: i + maxlen])
        next_word.append(x_test[i + maxlen])


X = np.zeros((len(bigrams), maxlen, vectorSize), dtype=np.bool)
y = np.zeros((len(next_word), vectorSize), dtype=np.bool)
for i, bigram in enumerate(bigrams):
    for t, word in enumerate(bigram):
        X[i, t, word] = 1
    y[i, next_word[i]] = 1


perplexity = 0
N = 0

for i in range(len(bigrams)):
    preds = model.predict(X[i], verbose=0)[0]
    next_word = y[i]
    prob = preds[next_word]
    #import pdb;pdb.set_trace()
    print (prob,i)
    perplexity += np.log(prob)
    N += 1

perplexity /= -N
print (perplexity)

exit(1)




print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
