from medpy.io import load
from medtools import *

def ContourTxtToPotlist(file):
  p_l = file.readlines()
  point_list = []
  for i in p_l:
  	i = i.strip('\n')
  	i = i.split()
  	x = float(i[1])
  	y = float(i[0])
  	x =int(round(x))
  	y = int(round(y))
  	# print x
  	point_list.append((x,y))
  return point_list
  	







train_path = '/media/dsigpu5/SSD/YUANHAN/data/miccai_lv/micca_LV/challenge_training/SC-N-2/'
contour_path = '/media/dsigpu5/SSD/YUANHAN/data/miccai_lv/micca_LV/train_label/TrainingDataContours/SC-N-02/contours-manual/IRCCI-expert/'
image_data, image_header = load(train_path + 'IM-0898-0100.dcm')
image_data = image_data.astype('float32')
F_i = open(contour_path + 'IM-0001-0100-icontour-manual.txt','r') 
F_o = open(contour_path + 'IM-0001-0100-ocontour-manual.txt','r') 
# print len(F.readlines())
print "image shape is:", image_data.shape
image = image_data/image_data.max()
print image.dtype
p_i = ContourTxtToPotlist(F_i)
p_m_i = PtToMap(p_i,image_data.shape)*0.1

p_o = ContourTxtToPotlist(F_o)
p_m_o = PtToMap(p_o,image_data.shape)*0.1

plt.imshow(image+p_m_i + p_m_o,cmap = 'gray',interpolation = 'nearest')
plt.show()

# print image_data.shape

