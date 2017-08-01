import numpy as np 
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter,polygon, circle
from skimage.filters import gaussian
from skimage.morphology import disk, dilation
from skimage.filters.rank import gradient
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import resize
from scipy.spatial import distance
from math import acos
from math import sqrt
from math import pi
from math import *
from scipy import ndimage
from skimage import measure
import skfmm
from sklearn.utils import shuffle
# from scipy.interpolate import splprep, splev
from scipy.interpolate import splprep, splev
patch_size = 32 #pathcsize * 2




def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1] * w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1] * w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v) * length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
   

def angle_clockwise(A,B = (0,-1)): #norm to angle (degree)
    if A == (0,0):
      return 0
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360 - inner

def corp(ar,angle,x,y,w=32): # ar = input image, w = patch size

  r = int(round(1/(np.sin(np.pi/4))*w,0)) + 1

  rr, cc = circle(y,x,r)

  wx,wy = ar.shape
  plain = np.zeros((wx,wy))

  plain[rr,cc] = 1
  roi = ar*plain
  roi = roi[y - r + 1:y + r ,x - r + 1:x + r ]
  # roi = roi[y-w:y+w,x-w:x+w]
  roi = rotate(roi,angle)
  # print roi.shape
  m_x, m_y = roi.shape
  m_x = int(round(m_x/2,0))
  m_y = int(round(m_y/2,0)) 
  f = roi[m_x-w:m_x+w ,m_y-w:m_y+w]
  # f = roi[m_x-w+1:m_x+w + 1,m_y-w+1:m_y+w + 1]
  # print f.shape 
  # f = resize(f,(2*w,2*w),preserve_range = True)
  # print f.max()
  return f




def generate_psedu_points(label,k = 15,ee = 10): #ee: how far is going to dialiated. k: distance between each point

  e = ee
  selem = disk(e)
  dilated_label = dilation(label, selem)
  dd_label = ndimage.binary_fill_holes(dilated_label).astype(int)
  contours_label = measure.find_contours(dd_label, 0.8)

  p_l = np.array(contours_label)
  p_l = p_l[0,:,:]
  
  p_l[:,[0,1]] = p_l[:,[1,0]]

  # print p_l

  keep_index_train = []
  for i in range(0,len(p_l),k):
    keep_index_train.append(i)
  p_l = p_l[keep_index_train]
  return p_l

def fill_holes(array):
  l = []
  for i in range(len(array)):
    image_c = array[i]
    image = image_c[:,:,0]
    image = ndimage.binary_fill_holes(image).astype(int)
    l.append(image)
  return np.array(l)

def rotate_vector(vec,an):#take degree(0-360) not radis(0-2pi), rotate single vector (x,y)

  ang = pi * an / 180
  x = vec[0]*cos(ang) - vec[1]*sin(ang)
  y = vec[0]*sin(ang) + vec[1]*cos(ang)
  return (x,y)

def rotate_vectors_list(vecs,an):#take degree(0-360) not radis(0-2pi), rotate a vecs list with an angel list!!!

  new_l = []
  for i,j in zip(vecs,an):
    j = pi * j / 180
    x = i[0]*cos(j) - i[1]*sin(j)
    y = i[0]*sin(j) + i[1]*cos(j)
    new_l.append([x,y])
  return np.array(new_l)
 
def generator_ponits_c(raw,k,xx = 128,yy = 128,r = 70):
  wx,wy = raw.shape
  init_points = []
  for x in range(0,k):
    x = float(x)
    d = (x/k)*2*np.pi
    x = np.cos(d)* (r + np.random.normal(0,5)) + xx
    y = np.sin(d)* (r + np.random.normal(0,5)) + yy
    init_points.append((int(round(x,0)),int(round(y,0))))
  return init_points

def PtToMap(i_p,size):
  i_m = np.zeros(size)
  for x in i_p:
    i_m[int(round(x[1]))][int(round(x[0]))] = 1 #used to be i_m[x[1]][x[0]] = 1
    #i_m[y,x]!!!
  return i_m
def PtOnMap(point,size): # single point on map
  i_m = np.zeros(size)
  i_m[int(round(point[1]))][int(round(point[0]))] = 1
  return i_m

def get_norm_T(points):
  ox = 128
  oy = 128
  norm_l = []
  for p in points:
    x = p[0]
    y = p[1]
    norm_l.append(angle_clockwise((p[0] - ox, p[1] - oy)))

  return norm_l



def get_vecF_from_label_abs(points,label): #less accuracy
  v_l = [0] * len(points)
  jj = 0
  for p in points:
    o_d = 1000
    for i in range(len(label)):#relative_y
      for j in range(len(label[i])):#relative_x
        if label[i][j] == 1:
          if distance.euclidean(p,(j,i)) < o_d:
            o_d = distance.euclidean(p,(j,i))
            v_l[jj] = (p[0] - j , p[1] - i)
    jj = jj + 1
  return v_l

def get_trajectory_first_derivitive(points):
  pt = np.array(points)
  tck, u = splprep(pt.T, u=None, s=0.0, per=1) 
  D1_mash = np.linspace(u.min(), u.max(), 1000)
  pass


def get_norm_by_spline_first_derivitive(points,angle =True):
  ps = points[:]
  ps.append(ps[-1])
  pt = np.array(ps)
  tck, u = splprep(pt.T, u=None, s=0.0, per=1) 
  u = u[:-1]


  D1_mash = np.linspace(u.min(), u.max(), 1000)
  xy = splev(u, tck, der=1) #produce dy and dx
  xy_T = np.array(xy).T

  norm_out = []
  norm_in = []
  
  for item in xy_T:

    norm_out.append(l2_norm((item[1],-item[0])))
    norm_in.append(l2_norm((-item[1],item[0])))

  return norm_in #size of 8
  
def normListToAngelList(norm):
  an_l = []
  for item in norm:
    an_l.append(angle_clockwise(item))
  return an_l

def normToAngel(norm):
  return angle_clockwise(norm)

def get_vecF_from_label_relative(points,label,ang):

  v_l = [0] * len(points)
  jj = 0
  # anan = get_norm_T(points)
  # ppp  = points
  # norm = get_norm_by_spline_first_derivitive(ppp)
  anan = ang
  p_l = []
  for i, j in zip(points,anan):
    x,y = i
    t_pat = corp(label,j,x,y,patch_size)
    p_l.append(t_pat)

  c_x = patch_size - 1
  c_y = patch_size - 1
  
  for pa in p_l:
    o_d = 10000.0
    for i in range(len(pa)):
      for j in range(len(pa[i])):
        if pa[i][j] > 0:
          if distance.euclidean((c_x,c_y),(j,i)) < o_d:
            o_d = distance.euclidean((c_x,c_y),(j,i))
            # v_l[jj] = (j-c_x,i-c_y)
            v_l[jj] = (c_x - j,c_y - i)
    jj = jj + 1
  return v_l 
  

     
def l2_norm(vec):
  dx,dy = vec
  mag = np.linalg.norm(vec)
  if mag == 0.0:
    return (1,0)
  return (dx/mag,dy/mag)

def l2_norm_list(vec_list): 
  pass

def mean_distance(vec_1,vec_2): #inputs are two vectors lists
  dis_l = []
  if vec_1.shape != vec_2.shape:
    print("vectors's shape mush be equal")
    return
  for i,j in zip(vec_1,vec_2):
    dis = (i[0] - j[0]) + (i[1] - j[1])**2
    dis_l.append(dis)
  return sum(dis_l)


def remove_empty_label(data,label):
  delet_list = []
  for i in range(len(label[:,:,:,0])):
    if np.sum(label[i,:,:,0]) == 0:
      delet_list.append(i)
  d = np.delete(data, delet_list,0)
  l = np.delete(label, delet_list,0)
  return d,l



def generate_mask(label,offset = 5):
  selem = disk(offset)
  filled_label = ndimage.binary_fill_holes(label).astype(int)
  diliated_filled_label = dilation(filled_label,selem)
  return diliated_filled_label



def get_SDMmap(img): #take l_train as input
  filled_reverted_l_train = 1 - ndimage.binary_fill_holes(img).astype(int) #[1,1,1,0,0,0,1,1,1]
  filled_reverted_l_train[filled_reverted_l_train == 0] = -1
  dis = skfmm.distance(filled_reverted_l_train,dx = 1)

  return dis, np.absolute(dis)



def get_gradient_SDMmap(SDMmap):
  x,y = SDMmap.shape

  temp = np.ndarray((x,y,2))
  kernel_x = np.array([[0,0,0],
                       [-1,0,1],
                       [0,0,0]])

  kernel_y = np.array([[0,-1,0],
                       [0,0,0],
                       [0,1,0]])

  dy = ndimage.convolve(SDMmap, kernel_y, mode='constant', cval=0.0)
  dx = ndimage.convolve(SDMmap, kernel_x, mode='constant', cval=0.0)

  temp = np.ndarray((x,y,2))
  temp[:,:,0] = dx
  temp[:,:,1] = dy
  
  return temp


def scale_SDMmap_gradient(SDM_gradient, scale_matrix):
  #use log(x+1) to bound: need to be added

  x,y = scale_matrix.shape

  xx,yy,zz = SDM_gradient.shape
  if x != xx or y != yy:
    print("function() scale_SDMmap_gradient shape not equal")

def iterate_mask(mask,random = 0.2):
  points_list = []
  for i in range(len(mask)):#iterate over Y row
    for j in range(len(mask[i])):#iterate over X 
      if mask[i][j] == 1:
        if np.random.uniform(0,1) <= random:
          points_list.append((j,i))
  return points_list



def multipy_sample_pathch_vecs(img,cord_xy,grad_corp_xy,grad_vec_xy,dev_list):
  x,y = cord_xy

  ang = angle_clockwise(l2_norm(grad_corp_xy))

  patch_list_per_an = []
  vec_list_per_an = []

  for dev in dev_list:
    angle = ang + dev
    patch = corp(img,angle,x,y)
    rotated_grad_vec = rotate_vector(grad_vec_xy, 360 - angle)
    # print rotated_grad_vec
    vec_list_per_an.append(rotated_grad_vec)
    patch_list_per_an.append(patch)
  return patch_list_per_an,vec_list_per_an



def corp_accdTo_mask(img,SDMmap_grad,SDM_vec_grad,mask_point_list):
  final_patch = []
  final_vec = []
  dev_list = [-45,0,45]
  print("the length of the mask_point_list is " + str(len(mask_point_list)))
  i = 0
  for cord in mask_point_list: #cord = (x,y)
    # print i
    i = i + 1
    x,y = cord
    dx = SDMmap_grad[y][x][0]
    dy = SDMmap_grad[y][x][1]

    dvx = SDM_vec_grad[y][x][0]
    dvy = SDM_vec_grad[y][x][1]

    gradient_corp = (dx,dy) 
    gradient_vec = (dvx,dvy) 

    

    patch_list_per_cord, vec_list_per_cord = multipy_sample_pathch_vecs(img,cord,gradient_corp,gradient_vec,dev_list)
    final_patch = final_patch + patch_list_per_cord
    final_vec = final_vec + vec_list_per_cord


  return np.array(final_patch), np.array(final_vec)

def get_limited_circle_gradient_SDMmap(label,ee=55):
  ssss = label.shape
  e = ee
  selem = disk(e)
  filled_label = ndimage.binary_fill_holes(label).astype(int)
  contours_label = measure.find_contours(filled_label, 0.8)
 
  p_l = np.array(contours_label)
  print("Contour's shape is",p_l.shape)
  p_l = p_l[0,:,:]

  p_l[:,[0,1]] = p_l[:,[1,0]] # rotate

  mask_label_contour = dilation(PtToMap(p_l,ssss), selem) ## size

  inner_mask = mask_label_contour*filled_label
  out_mask =  mask_label_contour - inner_mask
  label_SDM, label_abs_SDM = get_SDMmap(label)
  gradient = get_gradient_SDMmap(label_abs_SDM)
  scale_label_abs_SDM = (mask_label_contour*label_abs_SDM)/(mask_label_contour*label_abs_SDM).max()

  for i in range(len(out_mask)):#iterate over Y row
    for j in range(len(out_mask[i])):#iterate over X 
      if out_mask[i,j] == 1:
        scale = scale_label_abs_SDM[i][j]
        scale_deg = (1-scale)*90.0
        y = gradient[i,j,1]
        x = gradient[i,j,0]
        rotated_xy = rotate_vector((x,y),scale_deg)
        gradient[i,j,1] = rotated_xy[1]
        gradient[i,j,0] = rotated_xy[0]
  for i in range(len(inner_mask)):#iterate over Y row
    for j in range(len(inner_mask[i])):#iterate over X 
      if inner_mask[i,j] == 1:
        scale = scale_label_abs_SDM[i][j]
        scale_deg = (1-scale)*-90.0
        y = gradient[i,j,1]
        x = gradient[i,j,0]
        rotated_xy = rotate_vector((x,y),scale_deg)
        gradient[i,j,1] = rotated_xy[1]
        gradient[i,j,0] = rotated_xy[0]
  return gradient



# a = np.array([[0,0,0,0,0],
#               [0,0,1,0,0],
#               [0,1,2,1,0],
#               [0,0,1,0,0],
#               [0,0,0,0,0]])

# g_a = get_gradient_SDMmap(a)
# print g_a.shape






























# circle_raw = np.ones((256,256))
# circle_label = np.zeros((256,256))


# rr, cc = circle(128,128,50)
# circle_raw[rr,cc] = 0 
# rr, cc = circle_perimeter(128,128,50)
# circle_raw = gaussian(circle_raw,sigma = 8)
# circle_label[rr,cc] = 1





# ini_p = generator_ponits_c(circle_raw,k=8)

# ini_m = PtToMap(ini_p)
# # dd = get_vecF_from_label_abs(ini_p,circle_label)

# norm = get_norm_by_spline_first_derivitive(ini_p)

# an = normToAngel(norm)
# dd = get_vecF_from_label_relative(ini_p,circle_label,an)




# patch_l = []
# for i,j in zip(ini_p,an):
#   x,y = i
#   pat = corp(circle_raw,j,x,y,patch_size)
#   patch_l.append(pat)







# # for i in patch_l:
# #   plt.imshow(i,cmap = 'gray',interpolation = 'nearest')
# #   plt.show()
# #   print i.shape

# patchs = np.array(patch_l)
# distances = np.array(dd)

# # for px,d in zip(patch_l,distances):
# #   plt.imshow(px,cmap = 'gray',interpolation = 'nearest')
# #   plt.show()
# #   print d
# print ini_p
# print dd
# print an
# # # print distances
# np.save('patches.npy', patchs)
# np.save('vecs.npy', distances)

# plt.imshow(circle_raw + ini_m,cmap = 'gray',interpolation = 'nearest')
# plt.show()