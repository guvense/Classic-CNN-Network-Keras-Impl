import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def Model(input_shape):

    
    X_input = Input(input_shape)
    
    X = Conv2D(1,(5,5), strides = (1,1), name = 'conv0')(X_input)
    X = AveragePooling2D((2,2), strides = (2,2),name = "avg_pool")(X)
    X = Conv2D(1,(5,5), strides = (1,1), name = 'conv1')(X)
    X = AveragePooling2D((2,2), strides = (2,2),name = "avg_pool1")(X)
    X = Flatten()(X)
    X = Dense(1, activation = 'softmax', name='fc')(X)
    model = Model(inputs= X_input, outputs = X, name = 'modelim')

    
    ### END CODE HERE ###
    
    return model

def main():

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T



    leNetModel = Model(X_train.shape[1:])
    leNetModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    leNetModel.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 100)
    preds = leNetModel.evaluate(x = X_test, y = Y_test)
    
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


