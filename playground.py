
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure




# a = np.array([[0,0,0,0,0,0,0],
# 	          [0,0,0,0,0,0,0],
# 	          [0,0,0,-1,0,0,0],
# 	          [0,0,-1,-1,-1,0,0],
# 	          [0,0,0,-1,0,0,0],
# 	          [0,0,0,0,0,0,0],
# 	          [0,0,0,0,0,0,0]])

# # exit()
# SDMmap_train = get_SDMmap(a)
# SDMmap_gradient = get_gradient_SDMmap(SDMmap_train)
# print SDMmap_gradient

print angle_clockwise((0,1))
