# raw images shoud be saved in "images" folder
image_folder = './images'

# final preprocessed images will be stored
extracted_folder = './extracted_images'

# to store model files
models = './models'

# to stroe graphs
graphs = './graphs'

# vertical and horizontal size to be used

image_size_vertical = 100
image_size_horizontal = 100

# number of epochs to train a model
epoch = 100

# batch size used to train a model
batch_size = 64

# data set split ratio
train_ratio = 0.6
test_ratio = 0.2
validation_ratio = 0.2

# input data shape, this will be updated
# accordingly in the code for GREY_SCALE
# or RGB images if used.
x_shape = ()

# type of channels
GREY = 1
RGB = 3

# this config represents the image fusion
# in vertical or horizontal way
vertical = "VERTICAL"
horizontal = "HORIZONTAL"

# number of classes, this will be updated
# in code
num_classes = 0

# labeling of classes, this will be updated
# in code
person_label = {}