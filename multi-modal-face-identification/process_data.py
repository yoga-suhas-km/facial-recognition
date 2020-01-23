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
import PIL
from PIL import Image
from tqdm import tqdm

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

def process(fusion,channel):
    x = []
    y = []
    
    required_size = (config.image_size_vertical,config.image_size_horizontal)
    
    if not os.path.exists(config.extracted_folder):
        os.mkdir(config.extracted_folder)
    else:
        shutil.rmtree(config.extracted_folder, ignore_errors=True)
        os.mkdir(config.extracted_folder)
    
    if os.path.exists(config.image_folder):
        path, dirs, file = next(os.walk(config.image_folder))
        
        for dir in tqdm(dirs):
            path_t, dirs_t, file_t = next(os.walk(os.path.join(config.image_folder,dir)))
            for dir_t in dirs_t:
                if dir_t.split('_')[1] == "FACE":
                    if not os.path.exists(os.path.join(config.image_folder,dir,"FACE_EXTRACTED")):
                        os.mkdir(os.path.join(config.image_folder,dir,"FACE_EXTRACTED"))
                    else:
                        shutil.rmtree(os.path.join(config.image_folder,dir,"FACE_EXTRACTED"), ignore_errors=True)
                        os.mkdir(os.path.join(config.image_folder,dir,"FACE_EXTRACTED"))
                    extract_and_save_face(os.path.join(config.image_folder,dir,dir_t), os.path.join(config.image_folder,dir,"FACE_EXTRACTED"), config.image_size_vertical, config.image_size_horizontal)
        
        for dir in tqdm(dirs):
            face = []
            ear = []
            perioccular = []
            fuse = []
            i = 0
            img_name_face = []
            
            path_t, dirs_t, file_t = next(os.walk(os.path.join(config.image_folder,dir)))
            for dir_t in dirs_t:
                path_i, dirs_i, file_i = next(os.walk(os.path.join(config.image_folder,dir,dir_t)))
                if dir_t.split('_')[1] == "PERI":
                    perioccular = [ PIL.Image.open(os.path.join(config.image_folder,dir,dir_t,i)) for i in file_i ]
                elif dir_t.split('_')[1] == "EAR":
                    ear = [ PIL.Image.open(os.path.join(config.image_folder,dir,dir_t,i)) for i in file_i ]
                elif dir_t.split('_')[1] == "EXTRACTED":
                    img_name_face = file_i
                    face = [ PIL.Image.open(os.path.join(config.image_folder,dir,dir_t,i)) for i in file_i ]                    

            
            for f in face:
                for e in ear:
                    for p in perioccular:
                        fuse.append(f)
                        fuse.append(e)
                        fuse.append(p)
                        min_shape = sorted( [(np.sum(i.size), i.size ) for i in fuse])[0][1]
                        if fusion == config.horizontal:
                            imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in fuse ) )
                        elif fusion == config.vertical:
                            imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in fuse ) )

                        imgs_comb = Image.fromarray( imgs_comb)
                        imgs_comb = imgs_comb.resize(required_size)
                        imgs_comb.save(config.extracted_folder+'\\'+img_name_face[0].split('_')[0]+"_"+str(i)+".jpg")
                        i += 1    
                        fuse.clear()
    
    images, labels, training_data = get_data_labels(config.extracted_folder, channel)
    
    random.shuffle(training_data)
        
    for image, label in tqdm(training_data):
        x.append(image)
        y.append(label)

    """
    pyplot.imshow(x[6])
    pyplot.show()        
    
    print(y[6])
    """
    
    if channel == config.GREY:
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, config.GREY)
    else :
        x = np.array(x).reshape(-1, config.image_size_vertical, config.image_size_horizontal, config.RGB)

    if channel == config.GREY:
        pickle_out = open("x_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.GREY),"wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
        
        pickle_out = open("y_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.GREY),"wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
    else:
        pickle_out = open("x_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.RGB),"wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
     
        pickle_out = open("y_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.RGB),"wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
    return x, y
    

def get_dataset(fusion,channel):
    x = []
    y = []
    
    path, dirs, file = next(os.walk("."))
    if "num_classes.pickle" and "person_label.pickle" in file:
        pickle_in = open("num_classes.pickle","rb")
        config.num_classes = pickle.load(pickle_in)

        pickle_in = open("person_label.pickle","rb")
        config.person_label = pickle.load(pickle_in)        
    else:
        config.num_classes, config.person_label = prepare_labels(config.image_folder)
        pickle_out = open("num_classes.pickle","wb")
        pickle.dump(config.num_classes, pickle_out)
        pickle_out.close()        
        
        pickle_out = open("person_label.pickle","wb")
        pickle.dump(config.person_label, pickle_out)
        pickle_out.close()
    
    if channel == config.GREY :
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, config.GREY)
        if "x_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.GREY) and "y_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.GREY) in file:
            pickle_in = open("x_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.GREY),"rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_grey_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, 
                            config.image_size_horizontal, config.GREY),"rb")
            y = pickle.load(pickle_in)
            return x, y
        else :
            x, y = process(fusion,channel)
            return x, y
    else:
        config.x_shape = (-1, config.image_size_vertical, config.image_size_horizontal, config.RGB)
        if "x_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.RGB) and "y_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, config.image_size_horizontal, config.RGB) in file:
            pickle_in = open("x_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, 
                                config.image_size_horizontal, config.RGB),"rb")
            x = pickle.load(pickle_in)

            pickle_in = open("y_rgb_{}_{}_{}_{}.pickle".format(fusion,config.image_size_vertical, 
                                config.image_size_horizontal, config.RGB),"rb")
            y = pickle.load(pickle_in)
            return x, y
        else:
            x, y = process(fusion,channel)
            return x,y



def train_test_validation_set_split(x, y, train_ratio, test_ratio, validation_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = (1 - train_ratio), random_state = random.randint(0, 100))
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state = random.randint(0, 100)) 
    return x_train, x_test, x_val, y_train, y_test, y_val  
  