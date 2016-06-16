# -*- coding: utf-8 -*-
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers.noise import GaussianDropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.datasets import load_svmlight_file
from keras.regularizers import WeightRegularizer


class ManifoldWeightRegularizer(WeightRegularizer):

    def __init__(self, m=0., **kwargs):
        self.m = K.cast_to_floatx(m)
        super(ManifoldWeightRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')

        regularized_loss = loss + K.sum(K.abs(self.p)) * self.l1
        regularized_loss += K.sum(K.square(self.p)) * self.l2
        #
        out_dim = self.p.shape.eval()[-1]
        diff_mat = np.eye(out_dim) - np.eye(out_dim, k=1)
        diff_mat[-1, -1] = 0
        d = K.variable(diff_mat)
        regularized_loss += K.sum(K.square(K.dot(self.p, d))) * self.m
        return K.in_train_phase(regularized_loss, loss)


# 讀檔

def read_data(file):
    x, y = load_svmlight_file(file, dtype=np.float32)
    return x.todense(), y - 1  # y要從0開始
train_X, train_y = read_data('Data/training_data_libsvm.txt')
test_X, test_y = read_data('Data/testing_data_libsvm.txt')
feat_dim = train_X.shape[-1]
num_class = np.max(train_y) + 1
print 'feat_dim=%d, num_class=%d' % (feat_dim, num_class)
#
model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=feat_dim, init='uniform'))
model.add(GaussianDropout(0.5))
model.add(Dense(num_class, activation='softmax', W_regularizer=ManifoldWeightRegularizer(m=0.1)))
model.compile(optimizer='Adadelta',
              loss='sparse_categorical_crossentropy',  # 因為label直接是class id
              metrics=['accuracy'])
mdlchk = ModelCheckpoint(filepath='weights.best.hdf5', save_best_only=True, monitor='val_acc')
model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=100, nb_epoch=200, verbose=2, callbacks=[mdlchk])  # starts training
model.load_weights('weights.best.hdf5')
model.compile(optimizer=SGD(lr=1e-7, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=1, nb_epoch=3, verbose=1, callbacks=[mdlchk])  # starts training
model.load_weights('weights.best.hdf5')
loss, acc = model.evaluate(test_X, test_y, batch_size=5000)
print "Loss=%.4f, ACC=%.4f" % (loss, acc)
