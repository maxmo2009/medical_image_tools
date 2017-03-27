
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import data as skidata
import scipy
from medtools import *
from PIL import Image, ImageDraw
from medpy.metric.binary import dc
from medpy.metric.binary import precision
from medpy.metric.binary import recall

sensitivity = list()
specificity = list()
ppv = list()
npv = list()
jaccard = list()
dice = list()

data_p = '/media/dsigpu5/SSD/YUANHAN/data'
data = np.load(data_p + '/data/SCD/data_45.npy').astype(np.float32)
label = np.load(data_p + '/data/SCD/label_45.npy').astype(np.int32)
outfile = open("results/SCD_snakes.txt", 'w')

for idx in range(0, 45):
    test_label = label[idx,:,:,0]
    test_data = data[idx,:,:,0]
    # test_data = skidata.text()
    ee=20
    init = generate_psedu_points(test_label, ee=ee)

    # s = np.linspace(0, 2*np.pi, 400)
    # x = 420 + 100*np.cos(s)
    # y = 100 + 100*np.sin(s)
    # init = np.array([x, y]).T
    snake = active_contour(gaussian(test_data, 1), init, alpha=1, beta=1, gamma=40)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(test_data)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, test_data.shape[1], test_data.shape[0], 0])
    fig.savefig("figures/snake_%4d.png" % (idx))

    # segmentation result
    segmt = np.reshape(np.around(snake), (-1,)).astype(int)
    segmt = list(segmt)
    img = Image.new('L', test_data.shape, 0)
    ImageDraw.Draw(img).polygon(segmt, outline=1)
    outline = np.array(img)


    # black magic 1
    segmt_fill = ndimage.binary_fill_holes(outline).astype(int)
    label_fill = ndimage.binary_fill_holes(test_label).astype(int)
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
    outfile.write("%4d %.4f %.4f %.4f %.4f %.4f %.4f\n" % (idx, sens, spec, pre, negapre, jacc, overlap))

    print("Index: %4d, Overlap: %.4f"%(idx, overlap))

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
outfile.close()
