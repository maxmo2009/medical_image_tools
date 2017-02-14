
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt


X, Y = np.meshgrid(np.linspace(-1,1,50), np.linspace(-1,1,50))
phi = (X+0.8)**2+(Y+0.8)**2 - 0.01

speed = 1+X**2+Y**2

# d = skfmm.distance(phi, dx=2.0/500, periodic=True)
CS = plt.contour(X, Y, phi)
print CS.collections[0]
# plt.imshow(d)

# plt.show()
exit()
data_p = '/media/dsigpu5/SSD/YUANHAN/data'
label = np.load(data_p + '/data/labels.npy').astype(np.int32)
data = np.load(data_p +  '/data/datas.npy').astype(np.float32)

d_train = data[8,:,:,0]
l_train = label[8,:,:,0]

label_SDM, label_abs_SDM = get_SDMmap(l_train)
SDMmap_vec_gradient = get_gradient_SDMmap(label_abs_SDM)

U = SDMmap_vec_gradient[:,:,0]
V = SDMmap_vec_gradient[:,:,1]
Y, X = np.mgrid[-1:1:288j, -1:1:288j] #generate mash

# U = -1 - X**2 + Y
# V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)
lw = 5*speed / speed.max()


fig0, ax0 = plt.subplots()
# strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=lw, cmap=plt.cm.autumn)
strm = ax0.streamplot(X, Y, U, V, color=U,density=3, linewidth=2, cmap=plt.cm.autumn)
# fig0.colorbar(strm.lines)

# ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)                                                     

# plt.imshow(label_abs_SDM)

plt.show()
