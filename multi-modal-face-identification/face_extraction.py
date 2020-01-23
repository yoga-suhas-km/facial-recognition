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
from tqdm import tqdm
from matplotlib import pyplot # image operation
from PIL import Image # for more images
from mtcnn.mtcnn import MTCNN # face extraction model

platform = []

def extract_face(path_to_read_imgs, path_to_save_extracted_face, img_size_vertical, img_size_horizontal):
    required_size = (img_size_vertical, img_size_horizontal)
    # load image from file
    pixels = pyplot.imread(path_to_read_imgs)
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
    image.save(path_to_save_extracted_face)



def extract_and_save_face(path_to_read_imgs, path_to_save_extracted_face, image_size_vertical, image_size_horizontal):
    path_t, dirs_t, file_t = next(os.walk(path_to_read_imgs))
    for file in file_t: 
        if file.endswith('.jpg'):
            abs_path = os.path.abspath(os.path.join(path_to_read_imgs, file))
            platform = sys.platform
            if "W" in platform:
                extract_face(abs_path, path_to_save_extracted_face+'\\'+file, image_size_vertical, image_size_horizontal)
            else:    
                extract_face(abs_path, path_to_save_extracted_face+'/'+file, image_size_vertical, image_size_horizontal)
                

