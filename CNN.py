import tensorflow as tf 
import tensorlayer as tl 
import numpy as np
from medtools import *
import matplotlib.pyplot as plt










# with tf.device('/gpu:0'):


patches = np.load('data/patches_SDM_train.npy').astype(np.float32)
vecs = np.load('data/vecs_SDM_train.npy').astype(np.float32)

test_patches = np.load('data/patches_test.npy').astype(np.float32)
test_vecs = np.load('data/vecs_test.npy').astype(np.float32)

patches = patches[:,:,:,np.newaxis]
test_patches = test_patches[:,:,:,np.newaxis]

print patches.shape
print vecs.shape


tf.set_random_seed(0)
tl.layers.set_name_reuse(True)

n,x,y,c = patches.shape

batch_size = 200

xi=tf.placeholder(tf.float32, shape=[None, x, y, 1])
y_=tf.placeholder(tf.float32, shape=[None, 2])


network=tl.layers.InputLayer(xi,name='input_layer')

conv1=tl.layers.Conv2dLayer(network,
                        act=tf.nn.relu,shape=[3,3,1,32],
                       strides=[1,1,1,1],
                       padding='SAME',
                       W_init=tf.truncated_normal_initializer(stddev=0.1),
                       b_init=tf.constant_initializer(value=0.1),
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

# print dense1.outputs
# y = tf.nn.softmax

sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(denseO.outputs)))
ce_with = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(denseO.outputs,y_))
qrdic = tf.reduce_sum(tf.square(y_ - denseO.outputs))

train_step = tf.train.AdamOptimizer().minimize(qrdic)
init=tf.initialize_all_variables()
sess.run(init)

test_qrd_l = []
ress = np.array((0,0))
for i in range(2000):
  total_loss = 0
 
  for X_train, y_train in tl.iterate.minibatches(patches, vecs, batch_size, shuffle=True):
    _, los = sess.run([train_step,qrdic],feed_dict={xi:X_train,y_:y_train})
    
    total_loss = total_loss + los
    # print 'This is label: ', y_train
    # print 'This is predict: ', res
    # print 'This is loss: ', los
  if i%10 == 0:
    
    print "EPOCH: " + str(i) + ":"
    print "The total lose is:" + str(total_loss)

  #   # res = sess.run(denseO.outputs,feed_dict={xi:test_patches,y_:test_vecs})
  #   ress = sess.run(denseO.outputs,feed_dict={xi:test_patches})
  #   print "This is prediction: "
  #   print ress
  #   # print res
  #   print "This is label: "
  #   print test_vecs
  #   print "mean distance is :"
  #   print mean_distance(ress,test_vecs)
  #   test_qrd_l.append(mean_distance(ress,test_vecs))


# np.save('results.npy', ress)
# plt.plot(test_qrd_l)
# plt.show()
# exit()
    # print llos
  # if total_loss <= 0.00001:
  # 	print 'TERMINATE AT'
  # 	print total_loss
  # 	exit()