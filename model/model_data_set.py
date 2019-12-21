import sys
import model_global_variable
from keras.preprocessing.image import ImageDataGenerator
import keras


# this need to be replaced with out datasets
# we need to study how cifar10 dataset is?
from keras.datasets import cifar10


def get_dataset():
    sys.stdout.write('Loading Dataset\n')
    sys.stdout.flush()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, y_train, X_test, y_test

def get_preprocessed_dataset():
    X_train, y_train, X_test, y_test = get_dataset()

    sys.stdout.write('Preprocessing Dataset\n\n')
    sys.stdout.flush()

    X_train = X_train.astype('float32') / model_global_variable.dtype_mult
    X_test = X_test.astype('float32') / model_global_variable.dtype_mult
    y_train = keras.utils.to_categorical(y_train, model_global_variable.num_classes)
    y_test = keras.utils.to_categorical(y_test, model_global_variable.num_classes)

    return X_train, y_train, X_test, y_test
    
def image_generator():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1, # randomly zoom in on images by (percentage as fraction)
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
    )