import tensorflow as tf 
import tensorlayer as tl 
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure






# exit()


saver = tf.train.Saver()

data = np.load('../data/datas.npy').astype(np.float32)
label = np.load('../data/labels.npy').astype(np.int32)


print "The shape of test pathes is:"
print data.shape
train_label = label[6,:,:,0]
test_label = label[5,:,:,0]

train_data = data[6,:,:,0]
test_data = data[5,:,:,0]

test_points = generate_psedu_points(test_label)

ini_m = PtToMap(test_points,test_label.shape)


plt.imshow(ini_m + test_label)
plt.show()





