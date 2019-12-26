import os
import sys
import shutil
import pickle
import numpy as np # for array operation 
from matplotlib import pyplot # image operation
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
import cv2
import config
from face_extraction import extract_and_save_face

def label_img(name):
    word_label = name.split('_')[0]
    if word_label in config.person_label:
        return (config.person_label[word_label])

def get_data_labels(extracted_path, channel):
    images = []
    labels = []
    platform = sys.platform
    path = os.listdir(extracted_path)
    for i,img in enumerate(path):
        label = label_img(img)
        labels.append(label)
        if channel == 1:
            if "W" in platform:
                image = cv2.imread(extracted_path+'\\'+img ,cv2.IMREAD_GRAYSCALE) 
            else:
                image = cv2.imread(extracted_path+'/'+img ,cv2.IMREAD_GRAYSCALE)         
        else:
            if "W" in platform:
                image = pyplot.imread(extracted_path+'\\'+img)
            else:
                image = pyplot.imread(extracted_path+'/'+img)

        images.append(image)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def prepare_labels(image_folder):
    num_classes = 0
    person_label = {}
    keys = []
    path, dirs, file = next(os.walk(image_folder))
    num_classes = len(dirs)
    for i, img in enumerate(dirs):
        keys.append(i)
    label = zip(dirs, keys)
    person_label = dict(label)

    return num_classes, person_label


def get_dataset(channel):
    
    if not os.path.exists(config.extracted_folder):
        os.mkdir(config.extracted_folder)
    else:
        shutil.rmtree(config.extracted_folder, ignore_errors=True)
        os.mkdir(config.extracted_folder)
    
    extract_and_save_face(config.image_folder, config.extracted_folder, config.image_size_vertical, config.image_size_horizontal)
    
    config.num_classes, config.person_label = prepare_labels(config.image_folder)
    images, labels = get_data_labels(config.extracted_folder, channel)

    #pyplot.imshow(images[0])
    #pyplot.show()
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = config.test_size_t, random_state = random.randint(0, 100))
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    if channel == 1:
        x_train = x_train.reshape(x_train.shape[0], config.image_size_vertical, config.image_size_horizontal, 1)
        x_test = x_test.reshape(x_test.shape[0], config.image_size_vertical, config.image_size_horizontal, 1)
    else :
        x_train = x_train.reshape(x_train.shape[0], config.image_size_vertical, config.image_size_horizontal, 3)
        x_test = x_test.reshape(x_test.shape[0], config.image_size_vertical, config.image_size_horizontal, 3)    
    
    y_train = np_utils.to_categorical(y_train, config.num_classes)
    y_test = np_utils.to_categorical(y_test, config.num_classes)
    
    if channel == 1:
        pickle_out = open("x_train_grey.pickle","wb")
        pickle.dump(x_train, pickle_out)
        pickle_out.close()
    
        pickle_out = open("x_test_grey.pickle","wb")
        pickle.dump(x_test, pickle_out)
        pickle_out.close()
    
        pickle_out = open("y_train_grey.pickle","wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
    
        pickle_out = open("y_test_grey.pickle","wb")
        pickle.dump(y_test, pickle_out)
        pickle_out.close()
    else:
        pickle_out = open("x_train_rgb.pickle","wb")
        pickle.dump(x_train, pickle_out)
        pickle_out.close()
    
        pickle_out = open("x_test_rgb.pickle","wb")
        pickle.dump(x_test, pickle_out)
        pickle_out.close()
    
        pickle_out = open("y_train_rgb.pickle","wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()
    
        pickle_out = open("y_test_rgb.pickle","wb")
        pickle.dump(y_test, pickle_out)
        pickle_out.close()
    
    return x_train, x_test, y_train, y_test
    

 # used to test
def main():
    x_train, x_test, y_train, y_test = get_dataset(3)


if __name__ == "__main__":
    # execute only if run as a script
    main()
