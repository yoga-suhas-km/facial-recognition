""" 
MIT License

Copyright (c) 2019 Yoga Suhas Kuruba Manjunath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import time
import os
import config
from matplotlib import pyplot # image operation
import sys
from process_data import get_dataset, train_test_validation_set_split
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import load_model


dense_layers = [1,2]
layer_sizes = [32, 64, 128, 256]
conv_layers = [2]

def model(x_train, x_test, x_valid, y_train, y_test, y_valid, channel):
    platform = sys.platform
   
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
                              
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
                
                mc = ModelCheckpoint(os.path.join(config.models, "{}.h5".format(NAME)), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                
                history = model.fit(x_train, y_train, verbose=1, 
                            epochs=config.epoch, batch_size=config.batch_size,
                             validation_data=(x_valid, y_valid),
                            callbacks=[tensorboard, es, mc])
                
                saved_model = load_model(os.path.join(config.models, "{}.h5".format(NAME)))
                
                train_scores = saved_model.evaluate(x_train, y_train, verbose=0)
                print("%s: %.2f%%" % ("train_"+saved_model.metrics_names[1], train_scores[1]*100))
                print("%s: %.2f%%" % ("train_"+saved_model.metrics_names[0], train_scores[0]*100))

                test_scores = saved_model.evaluate(x_test, y_test, verbose=0)
                print("%s: %.2f%%" % ("test_"+saved_model.metrics_names[1], test_scores[1]*100))
                print("%s: %.2f%%" % ("test_"+saved_model.metrics_names[0], test_scores[0]*100))

                y_pred_t = saved_model.predict(x_test, batch_size=config.batch_size, verbose=0)
                y_pred_bool = np.argmax(y_pred_t, axis=1)
                y_pred = saved_model.predict_classes(x_test)
                
                print("PREDICTED-DATA")
                print(y_pred)
                
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                #pyplot.show()
                pyplot.savefig(os.path.join(config.graphs, "{}.png".format(NAME)))
                
def print_notice():
    print("please mention:")
    print("     fusion pattern: v for Vertical or h for Horizontal")
    print("     Channel: g for GREY or r for RGB")
    print("please try any of the following commands:")
    print("     python model.py v g")
    print("     python model.py v r")
    print("     python model.py h g")
    print("     python model.py h r")

def main(argv):

    if (len(argv) < 2) or (argv[0] != "v" and argv[0] !="h") or (argv[1] != "g" and argv[1] != "r"):
        print_notice()
        sys.exit()
    else:
        if argv[0] == "v":
            fusion_pattren = config.vertical
        elif argv[0] == "h":
            fusion_pattren = config.horizontal
        else:
            print_notice()
            sys.exit()
            
        if argv[1] == "r":
            channel = config.RGB
        elif argv[1] == "g":
             channel = config.GREY
        else:
            print_notice()
            sys.exit()

            
    x, y = get_dataset(fusion_pattren, channel)

    x_train, x_test, x_valid, y_train, y_test, y_valid = train_test_validation_set_split(x,y, config.train_ratio, config.test_ratio, config.validation_ratio)

    print("NUMBER OF PERSONS")
    print(config.num_classes)
    x_train = x_train/255.0
    x_test = x_test/255.0
    x_valid = x_valid/255.0
    
    print("TEST_DATA")
    print(y_test)
    
    y_train = np_utils.to_categorical(y_train, config.num_classes)
    y_test = np_utils.to_categorical(y_test, config.num_classes)
    y_valid = np_utils.to_categorical(y_valid, config.num_classes)
    
    if not os.path.exists(config.models):
        os.mkdir(config.models)    
    
    if not os.path.exists(config.graphs):
        os.mkdir(config.graphs)
        
    model(x_train, x_test, x_valid, y_train, y_test, y_valid, channel)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])