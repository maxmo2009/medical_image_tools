
# greedy based Active Contour Model
# 

import numpy as np 
import matplotlib.pyplot as plt
from skimage.draw import circle,polygon
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

circle_np = np.ones((256,256))
noise = np.random.normal(0.5,0.1,(256, 256))
init_r = 90
K = 20
init_points = []



img_as = data.astronaut()
img_as = rgb2gray(img_as)


for x in range(0,K):
  x = float(x)
  d = (x/K)*2*np.pi
  x = np.cos(d)* (init_r+np.random.normal(0,0.1)) + 128
  y = np.sin(d)* (init_r+np.random.normal(0,0.1)) + 128

  init_points.append((int(x),int(y)))

# init_map = np.zeros((512,512))


def PtToMap(i_p):
  i_m = np.zeros((256,256))
  for x in i_p:
    i_m[x[0]][x[1]] = 1
  return i_m

init_map = PtToMap(init_points)


# r = np.array([150, 170, 180, 110])
# c = np.array([150, 70, 200, 210])
# rr, cc = polygon(r, c)


rr, cc = circle(128,128,60)
circle_np[rr,cc] = 0
circle_np = gaussian(circle_np,sigma = 5)
circle_np = circle_np + noise

circle_np = (circle_np/circle_np.max())

img_g = gradient(circle_np,disk(20))
gx,gy = np.gradient(img_as)

abs_g = np.sqrt((gx**2) + (gy**2))

# plt.imshow(circle_np + init_map,cmap = 'gray',interpolation = 'nearest')

# plt.show()
plt.imshow(img_g,cmap = 'gray',interpolation = 'nearest')
plt.show()




def mean_distance(l):# Average Distance, l is a tuple list
  su_m = 0
  mean_d = 0
  for i in range(len(l)-1):
    d = np.sqrt((l[i+1][0] - l[i][0])**2 +  (l[i+1][1] - l[i][1])**2)
    su_m = su_m + d
    mean_d = su_m/(len(l)-1)
  return mean_d

def Int_E_count(l):
  ecout_i = []
  m_d = mean_distance(l)
  
  for i in range(len(l)-1):
    s = m_d - np.sqrt((l[i+1][0] - l[i][0])**2 + (l[i+1][1] - l[i][1])**2)
    # s = np.sqrt((l[i+1][0] - l[i][0])**2 + (l[i+1][1] - l[i][1])**2)
    a = s**2
    ecout_i.append(a)
  return sum(ecout_i)


def Int_E_curve(l):
  length = len(l)
  beta = np.ones(length) #for coners elimitaion

  ecurve = []
  # print length
  for i in range(1,length-1):
    s = (l[i-1][0] - 2*l[i][0] + l[i+1][0])**2 + (l[i-1][1] - 2*l[i][1] + l[i+1][1])**2
    ecurve.append(s)
  return sum(ecurve)

def Image_E(l,img_g):
  g = []
  for i in l:
  	grd = -img_g[i[0]][i[1]]
  	g.append(grd)
  return sum(g)

M_M = 4

def E_function(l,img_g,al=0.001,bet=0.4,gma=100):
  # print "Int_Count: " + str(Int_E_count(l))
  # print "Int_Curve: " + str(Int_E_curve(l))
  # print "Int_Image: " + str(Image_E(l,img_g))
  E = al*Int_E_count(l) + bet*Int_E_curve(l) + gma*Image_E(l,img_g)

  return E

# print E_function(init_points,abs_g)

def get_subpix(img,x,y,M):
  return img[y-M:y+M+1,x-M:x+M+1]

def find_lowest_cost(patch,curt_pt_i):

  temp_init_pt = init_points[:]
  # print type(init_points)
  # print init_points
  cp = init_points[curt_pt_i]

  max_cost = E_function(init_points,img_g)
  for i in range(len(patch)):#relative_y
    for j in range(len(patch[i])):#relative_x
      relative_o_p = (cp[0] - M_M, cp[1] - M_M)
      abs_pt = (relative_o_p[0] + j,relative_o_p[1] + i)
      temp_init_pt[curt_pt_i] = abs_pt
      if E_function(temp_init_pt,img_g) < max_cost:
        max_cost = E_function(temp_init_pt,img_g)
        init_points[curt_pt_i] = temp_init_pt[curt_pt_i]
        

      # print E_function(temp_ini_pt,img_g)
    # print patch[i]



plt.imshow(init_map + circle_np,cmap = 'gray',interpolation = 'nearest')
plt.show()
plt.close()


new = 0
while new != E_function(init_points,img_g):
# for g in range(100):
  print new
  new = E_function(init_points,img_g)
  
  for i in range(len(init_points)):
    x,y = init_points[i]
    sub_patch = get_subpix(img_g,x,y,M_M)
    
    # print "points: " + str(x) +  ' ' +str(y)
    # print sub_patch
    find_lowest_cost(sub_patch,i)
  


new_map = PtToMap(init_points)
plt.imshow(new_map + circle_np,cmap = 'gray',interpolation = 'nearest')
plt.show()







# a = np.array([[1, 2, 3, 4],
# 	          [4, 5, 6, 7],
# 	          [7, 8, 9, 8],
# 	          [7, 8, 9, 8]])
# w = 2
# print get_subpix(a,0,3,0)



