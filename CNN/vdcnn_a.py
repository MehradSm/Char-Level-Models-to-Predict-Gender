#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:17:38 2017

@author: drew
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:29:44 2017

@author: drew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:10:12 2017

@author: drew
"""

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, MaxPooling1D, Conv1D, Conv2D, MaxPooling2D, Dropout
#from keras.layers import LSTM, Lambda
from keras.layers import Embedding, BatchNormalization, merge
from keras.layers.core import Reshape
from keras.utils import plot_model, np_utils
import numpy as np
import tensorflow as tf
#from keras import backend as K
import keras.callbacks
import sys, pickle, re

#import tf.contrib.keras.layers.Lambda

def maxk(X):
    return tf.nn.top_k(X,8)

def maxk_outshape(in_shape):
    #in_shape[-1]= in_shape[-1]/8
    return in_shape[0],in_shape[1], in_shape[2]/8

def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)

def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None#'checkpoints/vdcnn2d_35_reviews.02-0.35.hdf5'

X = load_obj('X_books') +1
y = np_utils.to_categorical(load_obj('y_books')-1)

N,D = X.shape
N_train = int(N*0.9)
X_train = X[0:N_train,:]
X_test = X[N_train:-1,:]
y_train = y[0:N_train,:]
y_test = y[N_train:-1,:]
del X,y



## Equalizing gender ratios
#I_f = np.where(y_test == 1)[0]
#I_m = np.where(y_test == 0)[0]
#I_f = I_f[0:len(I_m)]
#X_test = np.append(X_test[I_f] , X_test[I_m], axis = 0)
#y_test = np.append(y_test[I_f] , y_test[I_m], axis = 0)
#del I_f , I_m

print(X_train.shape,y_train.shape)

[N,maxlen] = X_train.shape
N_test = len(y_test)

def cnn_block(in_layer, filters=64, k_size = 3, base_name = 'conv', resnet = False, b_mode = 'valid'):
    block = in_layer
    block = Conv1D(filters = filters, kernel_size = k_size, activation = None, name = base_name + str(1),padding = b_mode)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)

    block = Conv1D(filters = filters, kernel_size = k_size, activation = None, name = base_name + str(2),padding = b_mode)(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    if(block.get_shape()[-1] == in_layer.get_shape()[-1] and resnet): #notice I commented this out with 'and'
        return merge([in_layer,block], mode = 'sum', name = base_name + "_merge")
    else: return block
    
def cnn_block2d(in_layer,filters=64, kernels= (3,3), base_name = 'cnn', resnet = True, pool = True, b_mode = 'same', sub_s = (1,1)):
    CNN1 = Conv2D(filters,kernels, padding = b_mode, name = base_name + "a")(in_layer)
    if (resnet and in_layer.get_shape()[-1] == CNN1.get_shape()[-1]):
        CNN1 = merge([in_layer,CNN1], mode = 'sum')
    CNN2 = Conv2D(filters,kernels, padding = b_mode ,name = base_name + "b")(CNN1)
    if resnet: CNN2 = merge([CNN1,CNN2], mode = 'sum')
    CNN3 = Conv2D(filters,kernels, padding = b_mode, name = base_name + "c", subsample = sub_s)(CNN2)
    if resnet:
        CNN3 = merge([CNN2,CNN3], mode = 'sum')
    if pool: CNN3 = MaxPooling2D(pool_size = (2,2))(CNN3)
    return BatchNormalization()(CNN3)
    

def pool_and_concatenate(shortcut, current):
    long_form = keras.layers.concatenate([shortcut,current], axis = -1)
    return MaxPooling1D(pool_size=2)(long_form)


in_stat = Input(shape=(maxlen, ), dtype='int32')
embedded_stat = Embedding(input_dim = np.max(X_train) + 1, output_dim = 16, name = 'embed16')(in_stat)
cnn1 = Conv1D(filters = 32, kernel_size = 3, activation = 'relu', border_mode = 'same', name = 'cnn1')(embedded_stat)#9,7,3 other option for k_size

CB32 = cnn_block(cnn1, filters = 32, base_name = 'cb32a', b_mode = 'same', resnet = True)
CB32 = cnn_block(CB32, filters = 32, base_name = 'cb32b', b_mode = 'same', resnet = True)
CB32 = cnn_block(CB32, filters = 32, base_name = 'cb32c', b_mode = 'same', resnet = True)
CB32 = MaxPooling1D(pool_size=2)(CB32)

CB64 = cnn_block(CB32, filters = 64, base_name = 'cb64a', b_mode = 'same', resnet = True)
CB64 = cnn_block(CB64, filters = 64, base_name = 'cb64b', b_mode = 'same', resnet = True)
CB64 = cnn_block(CB64, filters = 64, base_name = 'cb64c', b_mode = 'same', resnet = False)
CB64 = MaxPooling1D(pool_size=2)(CB64)

CB64 = cnn_block(CB64, filters = 64, base_name = 'cb64d', b_mode = 'same', resnet = True)
CB64 = cnn_block(CB64, filters = 64, base_name = 'cb64e', b_mode = 'same', resnet = True)
CB64 = cnn_block(CB64, filters = 64, base_name = 'cb64f', b_mode = 'same', resnet = False)
CB64 = MaxPooling1D(pool_size=2)(CB64)

CB128 = cnn_block(CB64, filters = 128, base_name = 'cb64g', b_mode = 'same', resnet = True)
CB128 = cnn_block(CB64, filters = 128, base_name = 'cb64h', b_mode = 'same', resnet = True)
CB128 = cnn_block(CB64, filters = 128, base_name = 'cb64i', b_mode = 'same', resnet = False)
CB128 = MaxPooling1D(pool_size=2)(CB128)

txt_img = Reshape((128,64,1))(CB128)

CNN = cnn_block2d(txt_img, 16,(12,2),'conv1', sub_s = (2,1), pool = False, resnet = False)
CNN = cnn_block2d(CNN, 32,(9,2),'conv2')
CNN = cnn_block2d(CNN, 64,(6,2),'conv3')
CNN = cnn_block2d(CNN, 64,(4,2),'conv5')
CNN = cnn_block2d(CNN, 64,(4,2),'conv6')
CNN = cnn_block2d(CNN, 128,(3,3),'conv7')
CNN = cnn_block2d(CNN, 128,(3,3),'conv8')

classifier = Flatten()(CNN)
classifier = Dense(512,activation = 'relu', name = 'class1')(classifier)
classifier = BatchNormalization()(classifier)

#gender_guess = Dense(1, activation='sigmoid',name = 'class4')(classifier)

logits = Dense(5,activation = 'softmax')(classifier)

model = Model(inputs=in_stat, outputs=logits)
model.summary()
plot_model(model, to_file='model.png', show_shapes = True)

if checkpoint:
    model.load_weights(checkpoint, by_name=True)

file_name = 'vdcnn2d_35_reviews'

check_cb = keras.callbacks.ModelCheckpoint('checkpoints/'+file_name+'.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
adam = keras.optimizers.adam(decay=1e-5)#4e-5
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) #binary_crossentropy
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 256, epochs=20, shuffle=True, callbacks=[check_cb, earlystop_cb])
save_obj(hist.history,file_name + "_hist")

#python vdcnn_2d.py checkpoints/vdcnn9.10-0.82.hdf5
