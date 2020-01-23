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
                

