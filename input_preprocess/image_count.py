import os


def count_images(count_images_in_folder):
    number_of_images = []
    
    path, dirs, files = next(os.walk(count_images_in_folder))
    num_classes = len(dirs)
    
    for i in files:
        if i.endswith('.jpg'):
            number_of_images.append(1)
    
    for i in  dirs:
        path, dirs, files = next(os.walk(os.path.join(count_images_in_folder, i)))
        for j in files:
            if j.endswith('.jpg'):
                number_of_images.append(1)

    file_count = len(number_of_images)
    return file_count
    
    