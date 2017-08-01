import tensorflow as tf 
import tensorlayer as tl 
import numpy as np
from medtools import *
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle

data_p = '/media/dsigpu5/SSD/YUANHAN/data'
model_n = 'scd_addon'
# 





patches = np.load(data_p + '/train_data/TMI/SCD_indvidual_0_30_compressed_patch.npy').astype(np.float32)
vecs = np.load(data_p + '/train_data/TMI/SCD_indvidual_0_30_compressed_vecs.npy').astype(np.float32)


print(patches.shape)
print(vecs.shape)


patches = patches[:,:,:,np.newaxis]




tf.set_random_seed(0)
tl.layers.set_name_reuse(True)

# n,x,y,c = (100,64,64,1)
n,x,y,c = patches.shape

batch_size = 512
xi=tf.placeholder(tf.float32, shape=[None, x, y, 1])
y_=tf.placeholder(tf.float32, shape=[None, 2])


network=tl.layers.InputLayer(xi,name='input_layer')

conv1=tl.layers.Conv2dLayer(network,
                        act=tf.nn.relu,shape=[3,3,1,32],
                       strides=[1,1,1,1],
                       padding='SAME',
                       W_init=tf.truncated_normal_initializer(stddev=0.1),
                       b_init=tf.constant_initializer(value=0.01),
                       name='conv1')

pool1=tl.layers.PoolLayer(conv1,ksize=[1, 2, 2 ,1],strides=[1,2,2,1],padding='SAME',pool=tf.nn.max_pool,name='pool1')



conv2=tl.layers.Conv2dLayer(pool1,
                        act=tf.nn.relu,shape=[3,3,32,64],
                       strides=[1,1,1,1],
                       padding='SAME',
                       W_init=tf.truncated_normal_initializer(stddev=0.1),
                       b_init=tf.constant_initializer(value=0.1),
                       name='conv2')

pool2=tl.layers.PoolLayer(conv2,ksize=[1, 2,2 ,1],strides=[1,2,2,1],padding='SAME',pool=tf.nn.max_pool,name='pool2')

conv3=tl.layers.Conv2dLayer(pool2,
                        act=tf.nn.relu,shape=[3,3,64,128],
                       strides=[1,1,1,1],
                       padding='SAME',
                       W_init=tf.truncated_normal_initializer(stddev=0.1),
                       b_init=tf.constant_initializer(value=0.1),
                       name='conv3')

pool3=tl.layers.PoolLayer(conv3,ksize=[1, 2,2 ,1],strides=[1,2,2,1],padding='SAME',pool=tf.nn.max_pool,name='pool3')

conv4=tl.layers.Conv2dLayer(pool3,
                        act=tf.nn.relu,shape=[3,3,128,256],
                       strides=[1,1,1,1],
                       padding='SAME',
                       W_init=tf.truncated_normal_initializer(stddev=0.1),
                       b_init=tf.constant_initializer(value=0.1),
                       name='conv4')
pool4=tl.layers.PoolLayer(conv4,ksize=[1, 2,2 ,1],strides=[1,2,2,1],padding='SAME',pool=tf.nn.max_pool,name='pool4')
flat1 = tl.layers.FlattenLayer(pool4, name='flatten_layer')
dense1=tl.layers.DenseLayer(flat1,
                     n_units=2048,
                     act = tf.nn.relu,
                     W_init=tf.truncated_normal_initializer(stddev=0.1),
                     name ='relu_layer'
                     )
denseO = tl.layers.DenseLayer(dense1,
                     n_units=2,
                     act = tf.identity,
                     W_init=tf.truncated_normal_initializer(stddev=0.1),
                     name ='output'
                     )



sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(denseO.outputs)))
# ce_with = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(denseO.outputs,y_))
qrdic = tf.reduce_sum(tf.square(y_ - denseO.outputs))

train_step = tf.train.AdamOptimizer().minimize(qrdic)
init=tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()


#################################################################################
start_time = time.time()
# saver.restore(sess, '../models/SCD_individual_1_3/temp/final_DEEP_SNAKE_1_miccai_limitedCircle')
for i in range(300):
  total_loss = 0

  for X_train, y_train in tl.iterate.minibatches(patches, vecs, batch_size, shuffle=True):
    _, los = sess.run([train_step,qrdic],feed_dict={xi:X_train,y_:y_train})

    # total_loss = total_loss + los
    # print 'This is label: ', y_train
    # print 'This is predict: ', res
    # print 'This is loss: ', los
  # if i%10 == 0:

  print("EPOCH: " + str(i) + ":")
  print("The total lose is:" + str(los))
  # if i%50 == 0:
  #   saver.save(sess, '../models/miccai_individual_1_3/temp/DEEP_SNAKE_' + model_n + '_at_' + str(los))
  # if los <= 0.5:
  #   break


# saver.save(sess, '../models/miccai_individual_1_3/final_DEEP_SNAKE_' + model_n)
tl.files.save_npz(denseO.all_params , name='SCD_model_compressed.npz')
elapsed_time = time.time() - start_time
print("time last for: ")
print(elapsed_time)


#####################################################################################
# ########################previous_gradient_single_point#############################

# saver.restore(sess, '../models/miccai_f30/final_DEEP_SNAKE_1_miccai_limitedCircle')
# # # ress = sess.run(denseO.outputs,feed_dict={xi:test_patches})
#
#
#
# data = np.load(data_p + '/data/datas_miccai_125.npy').astype(np.float32)
# label = np.load(data_p + '/data/labels_miccai_125.npy').astype(np.int32)
# data,label = shuffle(data,label,random_state=3)
# print "The shape of test pathes is:"
# print data.shape
#
# # train_label = label[6,:,:,0]
# test_label = label[35,:,:,0]
#
# # train_data = data[6,:,:,0]
# test_data = data[35,:,:,0]
#
#
# test_points = generate_psedu_points(test_label)
# single_point = test_points[5]
# print "the length of contour points is:"
# print len(test_points)
# # norm_list = get_norm_by_spline_first_derivitive(list(test_points))
# # an_list = normListToAngelList(norm_list)
# SDMmap_vec_gradient = get_limited_circle_gradient_SDMmap(test_label)
# x,y = single_point
# u = SDMmap_vec_gradient[y,x,0]
# v = SDMmap_vec_gradient[y,x,1]
# init_an = normToAngel((u,v))
#
# plt.imshow(test_label,cmap = 'gray',interpolation = 'nearest')
# plt.show()
#
# point_list = []
# angle = 0
# for i in range(300):
#
#
#   if i == 0:
#     print "first iteration"
#     xx,yy = single_point
#     patch = corp(test_data,init_an,xx,yy)
#     # plt.imshow(patch,cmap = 'gray',interpolation = 'nearest')
#     # plt.show()
#     patch = patch[np.newaxis,:,:,np.newaxis]
#     ress = sess.run(denseO.outputs,feed_dict={xi:patch})
#     print 'rel ang:', ress
#     abs_vec = rotate_vector(ress[0],init_an)
#     print single_point
#     single_point = single_point + abs_vec
#
#     abs_vec_norm = l2_norm(abs_vec)
#     angle = normToAngel(abs_vec_norm)
#     print 'abs ang:', angle
#     # p_m = PtOnMap(single_point,test_label.shape)
#     point_list.append(single_point)
#     continue
#     # plt.imshow(p_m + test_label,cmap = 'gray',interpolation = 'nearest')
#     # plt.show()
#   print "iteration", i
#   xx,yy = single_point
#   patch = corp(test_data,angle,xx,yy)
#   # plt.imshow(patch,cmap = 'gray',interpolation = 'nearest')
#   # plt.show()
#   patch = patch[np.newaxis,:,:,np.newaxis]
#
#   ress = sess.run(denseO.outputs,feed_dict={xi:patch})
#   print 'rel ang:', ress
#   abs_vec = rotate_vector(ress[0],angle)
#
#   single_point = single_point + abs_vec
#
#   abs_vec_norm = l2_norm(abs_vec)
#
#   angle = normToAngel(abs_vec_norm)
#   print 'abs ang:', angle
#   # p_m = PtOnMap(single_point,test_label.shape)
#
#   point_list.append(single_point)
#
# p_mm = PtToMap(point_list,test_label.shape)
# plt.imshow(test_data,cmap = 'gray',interpolation = 'nearest')
# plt.show()
# plt.imshow(p_mm*0.5 + test_data,cmap = 'gray',interpolation = 'nearest')
# plt.show()
#

