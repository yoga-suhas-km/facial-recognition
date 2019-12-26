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
from config import image_folder, extracted_folder, image_size_vertical
from config import image_size_horizontal, person_label, test_size_t, num_classes
from face_extraction import extract_and_save_face

def label_img(name):
    word_label = name.split('_')[0]
    if word_label in person_label:
        return (person_label[word_label])

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


def get_dataset(channel):
    if not os.path.exists(extracted_folder):
        os.mkdir(extracted_folder)
    else:
        shutil.rmtree(extracted_folder, ignore_errors=True)
        os.mkdir(extracted_folder)
    
    extract_and_save_face(image_folder, extracted_folder, image_size_vertical, image_size_horizontal)

    images, labels = get_data_labels(extracted_folder, channel)

    #pyplot.imshow(images[0])
    #pyplot.show()
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = test_size_t, random_state = random.randint(0, 100))
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    if channel == 1:
        x_train = x_train.reshape(x_train.shape[0], image_size_vertical, image_size_horizontal, 1)
        x_test = x_test.reshape(x_test.shape[0], image_size_vertical, image_size_horizontal, 1)
    else :
        x_train = x_train.reshape(x_train.shape[0], image_size_vertical, image_size_horizontal, 3)
        x_test = x_test.reshape(x_test.shape[0], image_size_vertical, image_size_horizontal, 3)    
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
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
