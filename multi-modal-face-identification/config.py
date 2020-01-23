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