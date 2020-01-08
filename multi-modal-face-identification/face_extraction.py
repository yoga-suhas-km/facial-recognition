import os
import sys
from progress.bar import Bar
from matplotlib import pyplot # image operation
from PIL import Image # for more images
from mtcnn.mtcnn import MTCNN # face extraction model
from image_count import count_images
from config import image_folder, extracted_folder, image_size_vertical, image_size_horizontal

bar = Bar('\r SAVING EXTRACTED FACES: ', max = count_images(str(image_folder)) )
platform = []


def extract_face(path_to_read_imgs, path_to_save_extracted_face, img_size_vertical, img_size_horizontal):
    required_size = (img_size_vertical, img_size_horizontal)
    # load image from file
    pixels = pyplot.imread(path_to_read_imgs)
    # create the detector, using default weights
    detector = MTCNN()
    #print(detector)
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    #print(results)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    #print(face)
    # resize pixels to the model size
    image = Image.fromarray(face)
    #print(image)
    image = image.resize(required_size)
    #print(image)
    image.save(path_to_save_extracted_face)



def extract_and_save_face(path_to_read_imgs, path_to_save_extracted_face, image_size_vertical, image_size_horizontal):
    for file_or_dir in os.listdir(path_to_read_imgs):
        abs_path = os.path.abspath(os.path.join(path_to_read_imgs, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            extract_and_save_face(abs_path, path_to_save_extracted_face, image_size_vertical, image_size_horizontal)
        else:                        # file
            #print("\r"+file_or_dir+"\n") # use this to debug if extraction fails
            bar.next()
            if file_or_dir.endswith('.jpg'):
                platform = sys.platform
                if "W" in platform:
                    extract_face(abs_path, path_to_save_extracted_face+'\\'+file_or_dir, image_size_vertical, image_size_horizontal)
                else:    
                    extract_face(abs_path, path_to_save_extracted_face+'/'+file_or_dir, image_size_vertical, image_size_horizontal)
                
    bar.finish()



""" # used to test
def main():
    extract_and_save_face(image_folder, extracted_folder, image_size_vertical, image_size_horizontal)


if __name__ == "__main__":
    # execute only if run as a script
    main()
"""