image_folder = './images'
extracted_folder = './extracted_images'
test_image_folder = './test_images'
test_extracted_folder= './test_extracted_images'

models = './models'
#models = '.'
#image_size_vertical = 224
#image_size_horizontal = 224
image_size_vertical = 100
image_size_horizontal = 100
epoch = 15
batch_size = 64
test_size_t = 0.3

x_shape = ()

GREY_SCALE = 1
RGB = 3

"""
num_classes = 4
person_label={  'AM' : 0,   
                'DT' : 1,
                'SRK' : 2,
                'WS' : 3
                }
"""
num_classes = 0
person_label = {}