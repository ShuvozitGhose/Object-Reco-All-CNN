from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import keras
import sys
print('python: {}'.format(sys.version))
print('keras: {}'.format(keras.__version__))

#load the data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#Lets determine the dataset characteristics
print('Training Images: {}'.format(x_train.shape))
print('Testing Images: {}'.format(x_test.shape))

# A single image
print(x_train[0].shape)

# create a grid of 3x3 images
for i in range(0,9):
	plt.subplot(330 + 1 + i)
	img = x_train[50 + i]
	plt.imshow(img)

#show the plot
plt.show()

# preprocessing the dataset

# fix random seed for reproducibility
seed=6
np.random.seed(seed)

#load the data
(x_train,y_train),(x_test,y_test) =cifar10.load_data()

#normalize the inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255.0
x_test /=255.0

#class label shape
print(y_train.shape)
print(y_train[0])

#hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]
print(num_class)

print(y_train.shape)
print(y_train[0])

#importing layers
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

def allcnn(weights = None):
    # define model type - Sequential
    model = Sequential()

    # add model layers
    model.add(Conv2D(96, (3,3), padding = 'same', input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(192, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(192, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1), padding = 'valid'))


    # add Global Average Pooling Layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    #load the weights
    if weights:
        model.load_weights(weights)

    # return the model
    return model

# define hyper parameter
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

#build model and pretrained weights
weights='all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

# define optimzer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

#print model summary
print(model.summary())


# define additional training parameters
#epochs = 350
#batch_size = 32


#fit the model
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
scores = model.evaluate(x_test, y_test, verbose = 1)
print('Accuracy: {}'.format(scores[1]))



