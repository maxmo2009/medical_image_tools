
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from medpy.io import load
from sklearn.utils import shuffle

from scipy.interpolate import splprep, splev,splrep


# def get_first_derivitat(pt):
#   tck, u = interpolate.splrep(pt.T[0], pt.T[1], s=0)
#   xy = splev(u, tck, der=1)
#   return l2_norm(np.array(xy).T[-1])






# a = np.array([[1,1,1,1],
#  	          [2,2,2,2],
# 	          [1,1,0,1],
#  	          [2,1,2,2]])
# b = np.array([[1,1],
#  	           [2,2],
#  	           [3,3],
#  	           [4,4],
#  	           [5,5]])



# tck= splrep(b.T[0], b.T[1], s=0)
# # D1_mash = np.linspace(u.min(), u.max(), 1000)
# xy = splev(, tck, der=1)
# print l2_norm(np.array(xy).T[-1])

# exit()
# plt.imshow(a)
# plt.show()
# exit()


# def get_limited_circle_gradient_SDMmap(label):
#   e = 40
#   selem = disk(e)
#   filled_label = ndimage.binary_fill_holes(label).astype(int)
#   contours_label = measure.find_contours(filled_label, 0.8)
 
#   p_l = np.array(contours_label)
#   p_l = p_l[0,:,:]

#   p_l[:,[0,1]] = p_l[:,[1,0]] # rotate

#   mask_label_contour = dilation(PtToMap(p_l,(288,288)), selem)

#   inner_mask = mask_label_contour*filled_label
#   out_mask =  mask_label_contour - inner_mask
#   label_SDM, label_abs_SDM = get_SDMmap(label)
#   gradient = get_gradient_SDMmap(label_abs_SDM)
#   scale_label_abs_SDM = (mask_label_contour*label_abs_SDM)/(mask_label_contour*label_abs_SDM).max()
#   print label_abs_SDM.max()
#   for i in range(len(out_mask)):#iterate over Y row
#     for j in range(len(out_mask[i])):#iterate over X 
#       if out_mask[i,j] == 1:
#         scale = scale_label_abs_SDM[i][j]
#         scale_deg = (1-scale)*90.0
#         y = gradient[i,j,1]
#         x = gradient[i,j,0]
#         rotated_xy = rotate_vector((x,y),scale_deg)
#         gradient[i,j,1] = rotated_xy[1]
#         gradient[i,j,0] = rotated_xy[0]
#   for i in range(len(inner_mask)):#iterate over Y row
#     for j in range(len(inner_mask[i])):#iterate over X 
#       if inner_mask[i,j] == 1:
#         scale = scale_label_abs_SDM[i][j]
#         scale_deg = (1-scale)*-90.0
#         y = gradient[i,j,1]
#         x = gradient[i,j,0]
#         rotated_xy = rotate_vector((x,y),scale_deg)
#         gradient[i,j,1] = rotated_xy[1]
#         gradient[i,j,0] = rotated_xy[0]
#   return gradient

# data_p = '/media/dsigpu5/SSD/YUANHAN/data'
# label = np.load(data_p + '/data/labels.npy').astype(np.int32)
# data = np.load(data_p +  '/data/datas.npy').astype(np.float32)

# single_data = data[8,:,:,0]
# single_lable = label[8,:,:,0]



# g = get_limited_circle_gradient_SDMmap(single_lable)
# U = g[:,:,0]
# V = g[:,:,1]
# Y, X = np.mgrid[-1:1:288j, -1:1:288j]

# # plt.imshow(get_limited_circle_gradient_SDMmap(single_lable))
# fig0, ax0 = plt.subplots()

# strm = ax0.streamplot(X, Y, U, V, color=U,density=15, linewidth=0.5, cmap=plt.cm.autumn)
# plt.show()


# X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
# XX, YY = np.meshgrid(np.arange(0, 2 * np.pi, .5), np.arange(0, 2 * np.pi, .5))

# U = np.cos(XX)
# V = np.sin(YY)
# print X.shape
# print U.shape

# plt.figure()
# plt.title('Arrows scale with plot width, not view')
# Q = plt.quiver(X, Y, U, V, units='width')
# qk = plt.quiverkey(Q, 1, 1, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')
# plt.show()
# exit()


path = "/media/dsigpu5/SSD/YUANHAN/data/miccai_lv/micca_LV/challenge_training/SC-HF-I-1/IM-0002-0009.dcm"
data_p = '/media/dsigpu5/SSD/YUANHAN/data'
label = np.load(data_p + '/data/labels_miccai_125.npy').astype(np.int32)
data = np.load(data_p +  '/data/datas_miccai_125.npy').astype(np.float32)


plt.imshow(data[32,:,:,0],cmap = 'gray',interpolation = 'nearest')
plt.show()

label_SDM, label_abs_SDM = get_SDMmap(label[32,:,:,0])
# SDMmap_vec_gradient = get_limited_circle_gradient_SDMmap(label[5,:,:,0])
SDMmap_vec_gradient = get_gradient_SDMmap(label_SDM)
Y, X = np.mgrid[0:288:36j, 0:288:36j]



U = SDMmap_vec_gradient[:,:,0]

V = SDMmap_vec_gradient[:,:,1]

U = U[::8,::8]/120
V = V[::8,::8]/120

print U.shape
print U

plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V,scale=1)
# qk = plt.quiverkey(Q, 1, 1, 2, r'$2 \frac{m}{s}$', labelpos='E',
                    # coordinates='figure')
plt.savefig('sdm.png')

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
strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=lw, cmap=plt.cm.autumn)
# fig0.colorbar(strm.lines)

# ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)                                                     

# plt.imshow(label_abs_SDM)

plt.show()
