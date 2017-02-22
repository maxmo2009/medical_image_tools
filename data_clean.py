
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

data_p = '/media/dsigpu5/SSD/YUANHAN/data'


label = np.load(data_p + '/data/labels.npy').astype(np.int32)
data = np.load(data_p +  '/data/datas.npy').astype(np.float32)
print "shape is"
print label.shape
print data.shape
data_non_zero, label_non_zero= remove_empty_label(data,label)
print "shape is:"
print data_non_zero.shape
print label_non_zero.shape

keep_list = []
for d in range(len(data_non_zero)):
  print "------current picture is:------", d
  t = ndimage.binary_fill_holes(label[d,:,:,0]).astype(int)
  if np.array_equal(t,label[d,:,:,0]) == False:
    print "This has been restored",d 
    keep_list.append(d)

ki = np.array(keep_list)  
cleaned_data = data_non_zero[ki]
cleaned_label = label_non_zero[ki]

print cleaned_label.shape
print cleaned_data.shape
# for d in range(len(cleaned_label)):
#   print "------current picture is:------", d
#   plt.imshow(cleaned_label[d,:,:,0])
#   plt.show()

inperfect_list = [12,13,14,24,25,16,26,27,28,39,40,41,42,51,52,60,61,68,69,70,95,96,97,98,99,100,101,102,119,120,133,149,150,151,161,163,192,193,194,195,196,206,207,208,209]

keep_list_2 = range(0,210)

ki2 = list(set(keep_list_2) - set(inperfect_list))

ki2 = np.array(ki2)
cleaned_data = cleaned_data[ki2]
cleaned_label = cleaned_label[ki2]
print cleaned_label.shape
print cleaned_data.shape
  
np.save(data_p + '/data/clean_datas.npy', cleaned_data)
np.save(data_p + '/data/clean_labels.npy', cleaned_label)





































