from medpy.io import load
from medtools import *
from scipy import misc
from os import listdir
from os.path import isfile, join
import glob
import os

def get_filenmame(name):
  name = name.split('.', 1)[0]
  return name
  	
def compare_ident(A,B):
  for a,b in zip(A,B):
    a = a.split('.', 1)[0]
    b = b.split('.', 1)[0]
    if a == b:
      print 1
    else:
      print a,b

def pick_corr(imagep,B):
  imagep = imagep.split('.', 1)[0]
  for label in B:
    lp = label.split('.', 1)[0]
    if lp == imagep:
      return lp + '.png'


data_p = '/media/dsigpu5/SSD/YUANHAN/data'
opath = '/media/dsigpu5/SSD/YUANHAN/data/CAP_data/subsetTraining_individual/set_1_3/'
# contour_path = '/media/dsigpu5/SSD/YUANHAN/data/miccai_lv/micca_LV/train_label/TrainingDataContours/SC-N-02/contours-manual/IRCCI-expert/'

ext_dcm = 'dcm'
ext_png = 'png'
os.chdir(opath)
result_dcm = [i for i in glob.glob('*.{}'.format(ext_dcm))]
result_png = [i for i in glob.glob('*.{}'.format(ext_png))]

label_list = []
data_list = []

print type(label_list)
c = 0
for image_path in result_dcm:
  label_path = pick_corr(image_path,result_png)
  image_data, image_header = load(opath+image_path)
  image_data = image_data.astype('float32')
  label_data = misc.imread(opath+label_path)
  label = (label_data[:,:,1]/label_data[:,:,1].max()).T
  
  label = label.astype('int32')
  image = image_data/image_data.max()
  
  label = resize(label,(288,288),preserve_range = True)
  image = resize(image,(288,288),preserve_range = True)

  # label = ndimage.binary_fill_holes(label).astype('int32')

  # plt.imshow(label,cmap = 'gray',interpolation = 'nearest')
  # plt.show()
  c = c+1
  print c
  label_list.append(label)
  data_list.append(image)
  
data_ = np.array(data_list)[:,:,:,np.newaxis]
label_ = np.array(label_list)[:,:,:,np.newaxis]

print data_.shape
print label_.shape 


np.save(data_p + '/data/miccai/data_set_1_3.npy', data_)
np.save(data_p + '/data/miccai/label_set_1_3.npy', label_)







# np.save(data_p + '/train_data/patchs_miccai_only_1.npy', train_patch)
# np.save(data_p + '/train_data/vecs_miccai_only_1.npy', train_vecs)