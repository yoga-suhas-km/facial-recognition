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
from image_count import count_images

def label_img(name):
    word_label = name.split('_')[0]
    if word_label in config.person_label:
        return (config.person_label[word_label])

def get_data_labels(extracted_path, channel):
    images = []
    labels = []
    training_data = []
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
        training_data.append([image,label])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, training_data

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

def process(channel):
    x = []
    y = []
    number_of_images_in_extracted_folder = 0
    
    number_of_images_in_images = count_images(config.image_folder)
    
    if os.path.exists(config.extracted_folder):
        number_of_images_in_extracted_folder = count_images(config.extracted_folder)
    
    if number_of_images_in_images != number_of_images_in_extracted_folder:
        if not os.path.exists(config.extracted_folder):
            os.mkdir(config.extracted_folder)
        else:
            shutil.rmtree(config.extracted_folder, ignore_errors=True)
            os.mkdir(config.extracted_folder)
        
        extract_and_save_face(config.image_folder, config.extracted_folder, config.image_size_vertical, config.image_size_horizontal)
   
    images, labels, training_data = get_data_labels(config.extracted_folder, channel)
       
    random.shuffle(training_data)
        
    for images, labels in training_data:
        x.append(images)
        y.append(labels)

        
    #pyplot.imshow(x[6])
    #pyplot.show()        
        
        
    if channel == 1:
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, 1)
    else :
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, 3)

    if channel == 1:
        pickle_out = open("x_grey.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
        
        pickle_out = open("y_grey.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
    else:
        pickle_out = open("x_rgb.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
     
        pickle_out = open("y_rgb.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
    return x, y
       
def process_test_data(channel):
    x = []
    y = []
    number_of_images_in_extracted_folder = 0
    
    number_of_images_in_images = count_images(config.test_image_folder)
    
    if os.path.exists(config.test_extracted_folder):
        number_of_images_in_extracted_folder = count_images(config.test_extracted_folder)
    
    if number_of_images_in_images != number_of_images_in_extracted_folder:
        if not os.path.exists(config.test_extracted_folder):
            os.mkdir(config.test_extracted_folder)
        else:
            shutil.rmtree(config.test_extracted_folder, ignore_errors=True)
            os.mkdir(config.test_extracted_folder)
        
        extract_and_save_face(config.test_image_folder, config.test_extracted_folder, config.image_size_vertical, config.image_size_horizontal)
   
    images, labels, test_data = get_data_labels(config.test_extracted_folder, channel)

    random.shuffle(test_data)
        
    for images, labels in test_data:
        x.append(images)
        y.append(labels)
        
    if channel == 1:
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, 1)
    else :
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, 3)

    if channel == 1:
        pickle_out = open("x_test_grey.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
        
        pickle_out = open("y_test_grey.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
    else:
        pickle_out = open("x_test_rgb.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
     
        pickle_out = open("y_test_rgb.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
    return x, y

def get_dataset(channel):
    x = []
    y = []
    
    path, dirs, file = next(os.walk("."))

    config.num_classes, config.person_label = prepare_labels(config.image_folder)
    print(config.num_classes)
    
    if channel == 1 :
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, 1)
        if "x_grey.pickle" and "y_grey.pickle" in file:
            pickle_in = open("x_grey.pickle","rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_grey.pickle","rb")
            y = pickle.load(pickle_in)
            return x, y
        else :
            x, y = process(channel)
            return x, y
    else:
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, 3)
        if "x_rgb.pickle" and "y_rgb.pickle" in file:
            pickle_in = open("x_rgb.pickle","rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_rgb.pickle","rb")
            y = pickle.load(pickle_in)
            return x, y
        else:
            x, y = process(channel)
            return x,y


def get_test_dataset(channel):
    x = []
    y = []
    
    path, dirs, file = next(os.walk("."))

    config.num_classes, config.person_label = prepare_labels(config.test_image_folder)
    print(config.num_classes)

    if channel == 1 :
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, 1)
        if "x_test_grey.pickle" and "y_test_grey.pickle" in file:
            pickle_in = open("x_test_grey.pickle","rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_test_grey.pickle","rb")
            y = pickle.load(pickle_in)
            return x, y
        else :
            x, y = process_test_data(channel)
            return x, y
    else:
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, 3)
        if "x_test_rgb.pickle" and "y_test_rgb.pickle" in file:
            pickle_in = open("x_test_rgb.pickle","rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_rgb.pickle","rb")
            y = pickle.load(pickle_in)
            return x, y
        else:
            x, y = process_test_data(channel)
            return x,y


"""
 # used to test
def main():
    x, y = get_dataset(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()
"""