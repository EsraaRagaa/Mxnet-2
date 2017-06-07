import mnistLoad as mst
import numpy as np
from PIL import Image

data_path = '/home/hades/mxnettest/mnist/'
train_data_name = 'train-images-idx3-ubyte'
train_label_name = 'train-labels-idx1-ubyte'
test_data_name = 't10k-images-idx3-ubyte'
test_label_name = 't10k-labels-idx1-ubyte'

train_data_reader = mst.Load_mnist_data(data_path+train_data_name)

Image_matrix = train_data_reader.getImage()

test_img = Image_matrix[0]
test_img = test_img.reshape((28,28))
img = Image.fromarray(np.uint8(test_img))

img.show()

train_label_reader = mst.Load_mnist_data(data_path+train_label_name)
label_vec = train_label_reader.getlable()
print label_vec
