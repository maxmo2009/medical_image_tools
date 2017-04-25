from medpy.io import load
from medtools import *
from scipy import misc
from os import listdir
from os.path import isfile, join
import glob
import os


def get_points(file):
  point_list = []
  file = file.readlines()
  for i in file:
    xy = i.rstrip()
    xy = xy.split()
    x = round(float(xy[0]))
    y = round(float(xy[1]))
    point_list.append((int(x),int(y)))
  return point_list

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
      return lp + '.txt'



ext_dcm = 'dcm'
ext_txt = 'txt'
data_o = 'sample_SCD/'

data_p = '/media/dsigpu5/SSD/YUANHAN/data'
os.chdir(data_o)
result_dcm = [i for i in glob.glob('*.{}'.format(ext_dcm))]
result_txt = [i for i in glob.glob('*.{}'.format(ext_txt))]


label_list = []
data_list = []

c = 0
for image_path in result_dcm:
  print image_path
  label_path = pick_corr(image_path,result_txt)
  image_data, image_header = load(image_path)
  image_data = image_data.astype('float32')
  label_data = get_points(open(label_path))
  label = PtToMap(label_data, image_data.shape)
  # label = (label_data[:,:,1]/label_data[:,:,1].max()).T
  
  label = label.astype('int32')
  image = image_data/image_data.max()
  
  label = label.T
  image = image

  label = ndimage.binary_fill_holes(label).astype('int32')

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
# plt.imshow(p_m*0.3+image_data,cmap = 'gray',interpolation = 'nearest')
# plt.show()
np.save(data_p + '/data/SCD/data_45.npy', data_)
np.save(data_p + '/data/SCD/label_45.npy', label_)