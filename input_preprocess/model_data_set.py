import os
import shutil
import glob # for file operation
import numpy as np # for array operation 
from numpy import expand_dims
from numpy import asarray
from matplotlib import pyplot # image operation
from PIL import Image # for more images
from mtcnn.mtcnn import MTCNN # face extraction model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
from keras.preprocessing.image import ImageDataGenerator
from model_global_variable import IMAGE_SIZE_V, IMAGE_SIZE_H, test_size_t, num_classes, person_label
#from keras import backend as K



image_path = r'.\Images'
extracted_path = r'.\extracted_images'

images = []
labels = []


def extract_face(filename, save_path, IMAGE_SIZE_V, IMAGE_SIZE_H):
    required_size = (IMAGE_SIZE_V,IMAGE_SIZE_H)
    # load image from file
    pixels = pyplot.imread(filename)

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    image.save(save_path)
    #face_array = asarray(image)
    #return face_array
    

def read_image(file_path, save_path):
    #face_array = 
    extract_face(file_path, save_path, IMAGE_SIZE_V, IMAGE_SIZE_H)
    #pyplot.imshow(face_array)
    #pyplot.show()
    #return face_array   
   
def extract_and_save_images(read_path, save_path):
    for file_or_dir in os.listdir(read_path):
        abs_path = os.path.abspath(os.path.join(read_path, file_or_dir))
        #print(abs_path)
        if os.path.isdir(abs_path):  # dir
            extract_and_save_images(abs_path, save_path)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                #print(abs_path)
                #print(save_path+'\\'+file_or_dir)
                read_image(abs_path, save_path+'\\'+file_or_dir)
                #images.append(image)
                #labels.append(file_or_dir)
                #print(labels)
                #print(len(labels))

def label_img(name):
    word_label = name.split('_')[0]
    if word_label in person_label:
        return (person_label[word_label])

def normalize_img(img):
    face_array = asarray(img)
    face_array = face_array.astype('float32')
    face_array = expand_dims(face_array, axis=0)
    return face_array

def get_data_labels(extracted_path):
    images = []
    labels = []
    
    path = os.listdir(extracted_path)
    for i,img in enumerate(path):
        label = label_img(img)
        #if label not in labels:
        #    labels.append(label)
        labels.append(label)
        image = pyplot.imread(extracted_path+'\\'+img)
        #image = normalize_img(face_array)
        images.append(image)
    
    #np.reshape(labels, [-1])
    #print(labels)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
 


def get_dataset():
    """
    if not os.path.exists(str(extracted_path)):
        os.mkdir(str(extracted_path))
    else:
        shutil.rmtree(extracted_path, ignore_errors=True)
        os.mkdir(str(extracted_path))
        
    extract_and_save_images(read_path = str(image_path), save_path = str(extracted_path))
    """
    images, labels = get_data_labels(str(extracted_path))
    np.reshape(labels, [-1])
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = test_size_t, random_state = random.randint(0, 100))
    #x_train = x_train.reshape(-1, IMAGE_SIZE_V, IMAGE_SIZE_H, 1)
    #x_test = x_test.reshape(-1, IMAGE_SIZE_V, IMAGE_SIZE_H, 1)
    #x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE_V, IMAGE_SIZE_H, 1)
    #x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE_V, IMAGE_SIZE_H, 1)
    #print(K.image_dim_ordering())
    x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE_V, IMAGE_SIZE_H, 3)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE_V, IMAGE_SIZE_H, 3)
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test

    #print(x_train)

#if __name__ == '__main__':
#    main()

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