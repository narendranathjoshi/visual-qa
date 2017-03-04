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


# load json and create model
json_file = open('model_character_gru.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_character_gru.h5")
print("Loaded model from disk")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for j in range(100):
    #test_sentence = test[j]
    #bigram = test_sentence.split()[:2]
    rint = random.randint(0,len(test)-maxlen)
    fgram = test[rint:rint+maxlen]
    #print ("Starting seed = " + str(bigram))
    next_char = None
    generated = "".join(fgram)
    i = 0
    while next_char != "</s>" and i < 100:
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(fgram):
            x[0, t, char_indices[char]] = 1.
        preds = loaded_model.predict(x, verbose=0)[0]
        #next_index = np.argmax(preds)
        next_index = sample(preds, random.uniform(0, 1))
        next_char = indices_char[next_index]
        generated += "" + next_char
        fgram = fgram[1:] + next_char
        i += 1
    print (generated)