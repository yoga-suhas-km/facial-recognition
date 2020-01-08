from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os
import random
import config
#from config import num_classes
from matplotlib import pyplot # image operation
import sys
from process_data import get_dataset
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

dense_layers = [1,2]
layer_sizes = [256]
conv_layers = [2]

def model(x_train, x_test, y_train, y_test, channel):
    platform = sys.platform
    
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}-{}".format(conv_layer, layer_size, dense_layer, int(time.time()), channel)
                print(NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=config.x_shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(config.num_classes))
                model.add(Activation('sigmoid'))

                if "w" in platform:
                    tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))
                else:
                    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
                    
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )
                model.fit(x_train, y_train, verbose=1, 
                            epochs=config.epoch, batch_size=config.batch_size, #config.batch_size, 
                             #validation_split=config.test_size_t,
                             validation_data=(x_test, y_test),
                            callbacks=[tensorboard])
                
                scores = model.evaluate(x_test, y_test, verbose=0)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                #model.save("{}.h5".format(NAME))
                model.save(os.path.join(config.models, "{}.h5".format(NAME)))


def main():
    x, y = get_dataset(config.GREY_SCALE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = config.test_size_t, random_state = random.randint(0, 100))
    
    print(x_train.shape)
    print(x_test.shape)
    print(config.num_classes)
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    y_train = np_utils.to_categorical(y_train, config.num_classes)
    y_test = np_utils.to_categorical(y_test, config.num_classes)
    print(y_train.shape)
    print(y_test.shape)
    #y = np.array(y)


    #print(type(x))
    #print(type(y))

    model(x_train, x_test, y_train, y_test,1)


if __name__ == "__main__":
    # execute only if run as a script
    main()