# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from nltk.tokenize import RegexpTokenizer
from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.constraints import unitnorm
from keras.regularizers import l1l2
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.utils.visualize_util import plot

class CosSim(Dense):
    def call(self, x, mask=None):
        nx = K.l2_normalize(x, axis=-1)
        nw = K.l2_normalize(self.W, axis=0)
        output = K.dot(nx, nw)
        if self.bias:
            output += self.b
        return self.activation(output)

def triletter(toks):
    result = []
    for word in [ '#%s#' % (w) for w in toks ]:
        result.append([ word[beg:beg+3] for beg in range(len(word)-2) ])
    return result
#http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    #('I love this berger.', 'pos'),
    #('I do not like this sandwich', 'neg'),
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]
#
tokenizer = RegexpTokenizer(r'\w+')
train_tri = [ (triletter(tokenizer.tokenize(txt.lower())),tag) for txt, tag in train ]
test_tri = [ (triletter(tokenizer.tokenize(txt.lower())),tag) for txt, tag in test ]
dictionary = set()
for word, tag in train_tri + test_tri:
    for tri in word:
        dictionary |= set(tri)
dictionary = list(dictionary)
max_len = max([ len(tri) for tri, tag in train_tri + test_tri ])

X_train = np.zeros((len(train), max_len, len(dictionary)), dtype=np.float32)
X_test = np.zeros((len(test), max_len, len(dictionary)), dtype=np.float32)
for row, (word, tag) in enumerate(train_tri):
    for n, w in enumerate(word):
        for tri in w:
            X_train[row, n, dictionary.index(tri)] = 1.0
for row, (word, tag) in enumerate(test_tri):
    for n, w in enumerate(word):
        for tri in w:
            X_test[row, n, dictionary.index(tri)] = 1.0

# Predictive model
y_train = np.zeros((len(train), 2), dtype=np.float32)
y_test = np.zeros((len(test), 2), dtype=np.float32)
for row, (tri, tag) in enumerate(train_tri):
    y_train[row, 1 if tag=='pos' else 0] = 1.0
for row, (tri, tag) in enumerate(test_tri):
    y_test[row, 1 if tag=='pos' else 0] = 1.0

# http://research.microsoft.com/apps/pubs/default.aspx?id=256230
model = Sequential()
model.add(Convolution1D(1000, 3, activation='tanh', W_regularizer=l1l2(l1=0.01, l2=0.01), b_regularizer=l1l2(l1=0.01, l2=0.01), init='uniform', border_mode='same', input_shape=(max_len,len(dictionary))))
model.add(MaxPooling1D(pool_length=max_len, border_mode='valid'))
model.add(Flatten())
model.add(Dense(300, activation='tanh', W_regularizer=l1l2(l1=0.01, l2=0.01), b_regularizer=l1l2(l1=0.01, l2=0.01)))
model.add(CosSim(2, activation='linear', bias=False, W_regularizer=l1l2(l1=0.01, l2=0.01)))
model.add(Activation(activation='softmax'))
model.compile(optimizer='Adagrad',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=2, nb_epoch=30)
loss, acc = model.evaluate(X_test, y_test)
#plot(model, to_file='cdssm_pred_model.png')
