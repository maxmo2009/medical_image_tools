import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medtools import *
import time
from sklearn.utils import shuffle
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from medpy.metric.binary import dc
from medpy.metric.binary import precision
from medpy.metric.binary import recall
from scipy import ndimage
from PIL import Image, ImageDraw
from skimage import measure
import pymongo
from pymongo import MongoClient
import gridfs
from datetime import datetime

client = MongoClient('146.169.33.34', 27020)
db = client.AdvanceGan
db.authenticate('xiaobai', 'xiaobai')
uuid = datetime.utcnow()
mfs = gridfs.GridFS(db, collection="MedicalCardcEvaluateImg")


data_p = '/media/dsigpu5/SSD/YUANHAN/data'
model_n = "miccai_1"
# model_n = '1_50_cleans_limitedCircle_PreSin_regula'
#

# with tf.device('/gpu:0'):

print("Loading patches")
# patches = np.load(data_p + '/train_data/patches_SDM_train_1_50_limitedCircle_preSin_shuffled.npy').astype(np.float32)
patches = np.load(data_p + '/train_data/patchs_miccai_only_1.npy').astype(np.float32)
print("Loading vecs")
# vecs = np.load(data_p + '/train_data/vecs_SDM_train_1_50_limitedCircle_preSin_shuffled.npy').astype(np.float32)
vecs = np.load(data_p + '/train_data/vecs_miccai_only_1.npy').astype(np.float32)
print("Loading data SUCCESS")

# test_patches = np.load(data_p + '/train_data/patches_test.npy').astype(np.float32)
# test_vecs = np.load(data_p + '/train_data/vecs_test.npy').astype(np.float32)

patches = patches[:,:,:,np.newaxis]
# test_patches = test_patches[:,:,:,np.newaxis]

# print (patches.shape)
print (vecs.shape)


tf.set_random_seed(0)
tl.layers.set_name_reuse(True)

n,x,y,c = patches.shape
# n,x,y,c = (791523,64,64,1)

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

# print dense1.outputs
# y = tf.nn.softmax

sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(denseO.outputs)))
ce_with = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(denseO.outputs,y_))
qrdic = tf.reduce_sum(tf.square(y_ - denseO.outputs))

train_step = tf.train.AdamOptimizer().minimize(qrdic)
init=tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
# regulizer
# for p in denseO.all_params:
#   qrdic = qrdic + tf.contrib.layers.l2_regularizer(0.001)(p)

'''
#################################################################################
start_time = time.time()
for i in range(2000):
    total_loss = 0

    for X_train, y_train in tl.iterate.minibatches(patches, vecs, batch_size, shuffle=True):
        _, los = sess.run([train_step,qrdic],feed_dict={xi:X_train,y_:y_train})

#     # total_loss = total_loss + los
#     # print 'This is label: ', y_train
#     # print 'This is predict: ', res
#     # print 'This is loss: ', los
#   # if i%10 == 0:

    print "EPOCH: " + str(i) + ":"
    print "The total lose is:" + str(los)
    # if i%50 == 0:
    #     saver.save(sess, '../models/presin_1_50_unreg/DEEP_SNAKE_' + model_n + '_at_' + str(los))
    if los <= 0.5:
        break


savepath = "/home/dsigpu5/Desktop/work_space/med_image_src/miccai_only_1"
saver.save(sess, savepath + '/final_DEEP_SNAKE_' + model_n)
elapsed_time = time.time() - start_time
print "time last for: "
print elapsed_time


#####################################################################################
# ########################previous_gradient_single_point#############################
'''

originpath = "/home/dsigpu5/Desktop/work_space/med_image_src"
saver.restore(sess, originpath + '/models/miccai_f30/final_DEEP_SNAKE_1_miccai_limitedCircle')
# # ress = sess.run(denseO.outputs,feed_dict={xi:test_patches})



data = np.load(data_p + '/data/datas_miccai_125.npy').astype(np.float32)
label = np.load(data_p + '/data/labels_miccai_125.npy').astype(np.int32)
data,label = shuffle(data,label,random_state=3)
print "The shape of teEvaluation 1: st pathes is:"
print data.shape

drop = 0


sensitivity = list()
specificity = list()
ppv = list()
npv = list()
jaccard = list()
dice = list()

for idx in range(30, data.shape[0]):
    try:
        test_label = label[idx,:,:,0]
        test_data = data[idx,:,:,0]


        test_points = generate_psedu_points(test_label)
        single_point = test_points[5]

        # norm_list = get_norm_by_spline_first_derivitive(list(test_points))
        # an_list = normListToAngelList(norm_list)
        SDMmap_vec_gradient = get_limited_circle_gradient_SDMmap(test_label)
        x,y = single_point
        u = SDMmap_vec_gradient[y,x,0]
        v = SDMmap_vec_gradient[y,x,1]
        init_an = normToAngel((u,v))

        point_list = []
        angle = 0

        for i in range(1000):
          if i % 100 == 0:
              print i

          if i == 0:
            # print "first iteration"
            xx,yy = single_point
            patch = corp(test_data,init_an,xx,yy)
            # plt.imshow(patch,cmap = 'gray',interpolation = 'nearest')
            # plt.show()
            patch = patch[np.newaxis,:,:,np.newaxis]
            ress = sess.run(denseO.outputs,feed_dict={xi:patch})
            # print 'rel ang:', ress
            abs_vec = rotate_vector(ress[0],init_an)
            # print single_point
            single_point = single_point + abs_vec

            abs_vec_norm = l2_norm(abs_vec)
            angle = normToAngel(abs_vec_norm)
            # print 'abs ang:', angle
            # p_m = PtOnMap(single_point,test_label.shape)
            point_list.append(single_point)
            continue
            # plt.imshow(p_m + test_label,cmap = 'gray',interpolation = 'nearest')
            # plt.show()
          # print "iteration", i
          xx,yy = single_point
          # black magic 3
          for p in range(0, 10):
              # print(test_data)
              # print(angle)
              # print(xx)
              # print(yy)
              patch = corp(test_data,angle,xx,yy)
              # plt.imshow(patch,cmap = 'gray',interpolation = 'nearest')
              # plt.show()
              patch = patch[np.newaxis,:,:,np.newaxis]

              ress = sess.run(denseO.outputs,feed_dict={xi:patch})
              # print 'rel ang:', ress
              abs_vec = rotate_vector(ress[0],angle)


              if i > 500:
                  # black magic 2
                  abs_vec_norm = l2_norm(abs_vec + get_first_der(point_list))
              else:
                  abs_vec_norm = l2_norm(abs_vec)

              angle = normToAngel(abs_vec_norm)
              # if i > 520:
              #    exit()
              # print 'abs ang:', angle
              # p_m = PtOnMap(single_point,test_label.shape)

          # if i % 30 == 0:
          #     plt.imshow(patch[0,:,:,0], cmap="gray")
          #     plt.savefig("figures/patch_%4d_%4d"%(idx, i))
          single_point = single_point + abs_vec
          point_list.append(single_point)

        label_fill = ndimage.binary_fill_holes(test_label).astype(int)

        # find point trajectory circle
        threshold = 3
        period = 250
        min = 1000000
        end = point_list[-1]
        for j in range(len(point_list)-30, -1, -1):
            cur = point_list[j]
            if abs(cur[0]-end[0]) <= threshold and abs(cur[1]-end[1]) <= threshold:
                dis = np.sum((cur-end)**2)
                if dis < min:
                    min = dis
                else:
                    period = len(point_list) - j
                    break



        # segmentation result
        segmt = np.reshape(np.around(point_list[-period:]), (-1,)).astype(int)
        segmt = list(segmt)
        img = Image.new('L', test_data.shape, 0)
        ImageDraw.Draw(img).polygon(segmt, outline=1)
        outline = np.array(img)


        # black magic 1
        segmt_fill = ndimage.binary_fill_holes(outline).astype(int)
        # segmt_fill2 = segmt_fill - outline
        # plt.imshow(segmt_fill2,cmap = 'gray',interpolation = 'nearest')
        # plt.savefig("figures/result_%4d_segmt.png" % (idx))


        ###########################################################################
        # Evaluation 1: sensitivity = recall
        sens = recall(label_fill, segmt_fill)
        sensitivity.append(sens)

        # Evaluation 2: specificity
        negative = label_fill.size - np.sum(label_fill)
        truenega = 0.
        for ii in range(label_fill.shape[0]):
            for ji in range(label_fill.shape[1]):
                if label_fill[ii,ji] == 0 and segmt_fill[ii, ji] == 0:
                    truenega += 1.
        spec = truenega / negative
        specificity.append(spec)

        # Evaluation 3: PPV = precision
        pre = precision(label_fill, segmt_fill)
        ppv.append(pre)

        # Evaluation 4: NPV
        negative_predict = segmt_fill.size - np.sum(segmt_fill)
        negapre = truenega / negative_predict
        npv.append(negapre)

        # Evaluation 5: Jaccard
        truepos = 0.
        for ii in range(label_fill.shape[0]):
            for ji in range(label_fill.shape[1]):
                if label_fill[ii,ji] == 1 and segmt_fill[ii, ji] == 1:
                    truepos += 1.
        jacc = truepos / (np.sum(label_fill) + np.sum(segmt_fill) - truepos)
        jaccard.append(jacc)

        # Evaluation 6: dice coefficient
        overlap = dc(label_fill, segmt_fill)
        dice.append(overlap)
        ###########################################################################

        print("Index: %4d, Overlap: %.4f"%(idx, overlap))


        ###########################################################################
        # save img
        red_label  = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        red_label[:,:,0] = contour(test_label)

        grey_data = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        for s in range(0, 3):
            grey_data[:,:,s] = test_data.copy()

        p_mm = PtToMap(point_list[-period:],test_label.shape)
        yellow_seg = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        for s in range(0, 3):
            yellow_seg[:,:,s] = p_mm.copy()

        img_data = grey_data*3 + red_label + yellow_seg
        print(np.max(img_data))
        plt.imshow(img_data/img_data.max())
        plt.savefig("figures/result_%4d_%.4f.png" % (idx, overlap))

        # red_label2  = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        # red_label2[:,:,0] = test_label
        # img_data2 = grey_data*3 + red_label2
        # plt.imshow(img_data2/img_data2.max())
        # plt.savefig("figures/result_%4d_label.png" % (idx))
        ###########################################################################


        ###########################################################################
        # save to mongodb
        current = datetime.utcnow()
        # labelimgid = mfs.put((img_data2/img_data2.max()).astype(np.float32).tostring(), filename="label-%s-%4d"%(repr(current), idx))
        resultimgid = mfs.put((img_data/img_data.max()).astype(np.float32).tostring(), filename="label-%s-%4d"%(repr(current), idx))
        dbresult = db.MedicalCardcEvaluateLog.insert_one({
            "uuid": uuid,
            "timestamp": current,
            "imgid": idx,
            # "labelimgid": labelimgid,
            "resultimgid": resultimgid,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": pre,
            "npv": negapre,
            "jaccard": jacc,
            "dc": overlap,
            })
        ###########################################################################


    except (IndexError):
        drop += 1
        print("Index: %4d, Drop: No.%2d"%(idx, drop))

        red_label  = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        print(test_label.shape)
        red_label[:,:,0] = contour(test_label)
        grey_data  = np.zeros((test_label.shape[0], test_label.shape[1], 3))
        for s in range(0, 3):
            grey_data[:,:,s] = test_data.copy()

        img_data = grey_data*5 + red_label
        print(np.max(img_data))
        plt.imshow(img_data/img_data.max())
        plt.savefig("figures/drop_%4d.png" % (idx))



print("p mean: \t %.4f" % (np.mean(sensitivity)))
print("p median: \t %.4f" % (np.median(sensitivity)))
print("q mean: \t %.4f" % (np.mean(specificity)))
print("q median: \t %.4f" % (np.median(specificity)))
print("PPV mean: \t %.4f" % (np.mean(ppv)))
print("PPV median: \t %.4f" % (np.median(ppv)))
print("NPV mean: \t %.4f" % (np.mean(npv)))
print("NPV median: \t %.4f" % (np.median(npv)))
print("J mean: \t %.4f" % (np.mean(jaccard)))
print("J median: \t %.4f" % (np.median(jaccard)))
print("D mean: \t %.4f" % (np.mean(dice)))
print("D median: \t %.4f" % (np.median(dice)))

# round1: without black magic 2: 0.8742, 0.8819, , 0.9184
# round2: with black magic 2:    0.8798, 0.8843, , 0.9215

# round3: new dataset: without black magic 2:  0.8866, 0.9199, , 0.9199
# round4: new dataset: with black magic 2   :  0.9557, 0.9366, , 0.9366

# round 5: 0.9557 0.9366 0.9558 0.9558
