import sys
import json
import model_global_variable, model_data_set
from pathlib import Path
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



def generate_optimizer():
    return keras.optimizers.Adam()

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])
				  
def generate_model():
    # check if model exists if exists then load model from saved state
    
    
    if Path('./models/convnet_improved_model.json').is_file():
        sys.stdout.write('Loading existing model\n\n')
        sys.stdout.flush()

        with open('./models/convnet_improved_model.json') as file:
            model = keras.models.model_from_json(json.load(file))
            file.close()

        # likewise for model weight, if exists load from saved state
        if Path('./models/convnet_improved_weights.h5').is_file():
            model.load_weights('./models/convnet_improved_weights.h5')

        compile_model(model)

        return model
    
    
    sys.stdout.write('Loading new model\n\n')
    sys.stdout.flush()

    model = Sequential()
    
    
    # Conv1 32 32 (32)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=model_global_variable.X_shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Conv2 16 16 (64)
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Conv2 8 8 (128)
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # below is vgg16 model
    """
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1',input_shape=model_global_variable.X_shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))
    """
    
    # FC
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(model_global_variable.num_classes))
    model.add(Activation('softmax'))

    # compile has to be done impurely
    compile_model(model)

    with open('./models/convnet_improved_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model
	
	
def train(model, X_train, y_train, X_test, y_test):
    sys.stdout.write('Training model with data augmentation\n\n')
    sys.stdout.flush()

    datagen = model_data_set.image_generator()
    datagen.fit(X_train)
    

    # train each iteration individually to back up current state
    # safety measure against potential crashes
    epoch_count = 0
    while epoch_count < model_global_variable.epoch:
        epoch_count += 1
        sys.stdout.write('Epoch count: ' + str(epoch_count) + '\n')
        sys.stdout.flush()
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=model_global_variable.batch_size),
                            steps_per_epoch=len(X_train) // model_global_variable.batch_size,
                            epochs=1,
                            validation_data=(X_test, y_test))
        sys.stdout.write('Epoch {} done, saving model to file\n\n'.format(epoch_count))
        sys.stdout.flush()
        model.save_weights('./models/convnet_improved_weights.h5')

    return model
	
	
def get_accuracy(pred, real):
    # reward algorithm
    result = pred.argmax(axis=1) == real.argmax(axis=1)
    return np.sum(result) / len(result)
    

def print_performance_metrics(model, X_test, y_test, batch_size):
    y_pred_t = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred_bool = np.argmax(y_pred_t, axis=1)
    y_pred = model.predict_classes(X_test)


    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred))
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred))
    print('F1 score: %f' % f1)
    
    print(classification_report(y_test, y_pred_bool,labels=np.unique(y_pred_bool)))