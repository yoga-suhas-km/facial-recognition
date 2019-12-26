import os


number_of_images = []

def count_images(count_images_in_folder):
    for file_or_dir in os.listdir(count_images_in_folder):
        abs_path = os.path.abspath(os.path.join(count_images_in_folder, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            count_images(abs_path)
        else:
            number_of_images.append(1)
            
    file_count = len(number_of_images)
    return file_count