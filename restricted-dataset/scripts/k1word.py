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


perplexity = 0
N = 0

for i in range(len(test)):
    w = test[i].split()
    for j in range(len(w)-maxlen):
        bigram = w[j:j+maxlen]
        x = np.zeros((1, maxlen, len(words)))
        for t, word in enumerate(bigram):
            x[0, t, word_indices[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_word = w[j+maxlen]
        prob = preds[word_indices[next_word]]
        #import pdb;pdb.set_trace()
        #print (prob,i)
        perplexity += np.log(prob)
        N += 1

perplexity /= -N
print (perplexity)

exit(1)
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 4):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=7, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
