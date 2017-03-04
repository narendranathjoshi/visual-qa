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


# load json and create model
json_file = open('model_word_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_word_lstm.h5")
print("Loaded model from disk")

perplexity = 0
N = 0

for i in range(len(test)):
    w = test[i].split()
    for j in range(len(w)-maxlen):
        bigram = w[j:j+maxlen]
        x = np.zeros((1, maxlen, len(words)))
        for t, word in enumerate(bigram):
            x[0, t, word_indices[word]] = 1.

        preds = loaded_model.predict(x, verbose=0)[0]
        next_word = w[j+maxlen]
        prob = preds[word_indices[next_word]]
        #import pdb;pdb.set_trace()
        #print (prob,i)
        perplexity += np.log(prob)
        N += 1

perplexity /= -N
perplexity = np.exp(perplexity)
print (perplexity)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for j in range(100):
    test_sentence = test[j]
    bigram = test_sentence.split()[:2]
    #print ("Starting seed = " + str(bigram))
    next_word = None
    generated = " ".join(bigram)
    i = 0
    while next_word != "</s>" and i < 100:
        x = np.zeros((1, maxlen, len(words)))
        for t, word in enumerate(bigram):
            x[0, t, word_indices[word]] = 1.
        preds = loaded_model.predict(x, verbose=0)[0]
        #next_index = np.argmax(preds)
        next_index = sample(preds, random.uniform(0.001, 1))
        next_word = indices_word[next_index]
        generated += " " + next_word
        bigram = bigram[1:] + [next_word]
        i += 1
    print (generated)