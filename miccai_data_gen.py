from medpy.io import load
from medtools import *
from scipy import misc


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
  	






data_p = '/media/dsigpu5/SSD/YUANHAN/data'
train_path = '/media/dsigpu5/SSD/YUANHAN/data/CAP_data/Training/DET0000201/'
label_path = '/media/dsigpu5/SSD/YUANHAN/data/CAP_data/Training/DET0000201/'
# contour_path = '/media/dsigpu5/SSD/YUANHAN/data/miccai_lv/micca_LV/train_label/TrainingDataContours/SC-N-02/contours-manual/IRCCI-expert/'
image_data, image_header = load(train_path + 'DET0000201_SA5_ph1.dcm')
image_data = image_data.astype('float32')
label = misc.imread(label_path + 'DET0000201_SA5_ph1.png')
# F_i = open(contour_path + 'IM-0001-0100-icontour-manual.txt','r') 
# F_o = open(contour_path + 'IM-0001-0100-ocontour-manual.txt','r') 
# # print len(F.readlines())
print "image shape is:", image_data.shape
print "label shape is:", label.shape
label = (label[:,:,1]/label[:,:,1].max()).T
image = image_data/image_data.max()
label = label.astype(int)

image = image[np.newaxis,:,:,np.newaxis]
label = label[np.newaxis,:,:,np.newaxis]
print "image shape is:", image_data.shape
print "label shape is:", label.shape
# p_i = ContourTxtToPotlist(F_i)
# p_m_i = PtToMap(p_i,image_data.shape)*0.1

# p_o = ContourTxtToPotlist(F_o)
# p_m_o = PtToMap(p_o,image_data.shape)*0.1

# plt.imshow(image + label,cmap = 'gray',interpolation = 'nearest')
# plt.show()
# np.save(data_p + '/miccai_traindata/data_1.npy', image)
# np.save(data_p + '/miccai_traindata/label_1.npy', label)
single_label = label[0,:,:,0]
single_data = image[0,:,:,0]
dialited_label_mask = generate_mask(single_label,offset = 15)
  # SDMmap_corp_gradient = get_gradient_SDMmap(SDMmap_corp_norm_train) #
SDMmap_corp_gradient = get_limited_circle_gradient_SDMmap(single_label)# previous Single point norm map
SDMmap_vec_gradient = get_limited_circle_gradient_SDMmap(single_label)
  

# plt.imshow(SDMmap_gradient,cmap = 'gray',interpolation = 'nearest')
# plt.show()


points_list =  iterate_mask(dialited_label_mask)

train_patch, train_vecs = corp_accdTo_mask(single_data,SDMmap_corp_gradient,SDMmap_vec_gradient,points_list)

# print image_data.shape

print train_patch.shape
print train_vecs.shape

np.save(data_p + '/train_data/patchs_miccai_only_1.npy', train_patch)
np.save(data_p + '/train_data/vecs_miccai_only_1.npy', train_vecs)