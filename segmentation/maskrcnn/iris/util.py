import cv2
import os
import re
import numpy as np
import os
import tempfile

from glob import glob

def load_images(im_name, gt_name, multi_channel_gt=False):
    # print('loading', im_name, gt_name)
    im = cv2.imread(im_name)
    gt = cv2.imread(gt_name, 0)

    if (im is None):
        print('Error loading image:', im_name)
        return im, im
    if (gt is None):
        gt = np.zeros((240, 320, 2), dtype='uint8')
    
    # make sure we don't mess up color images
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # handle different sizes
    if im.shape[:2]!=(240, 320):
        im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)

    if gt.shape[:2]!=(240, 320):
        gt = cv2.resize(gt, (320, 240), interpolation=cv2.INTER_AREA)

    if multi_channel_gt:
        # make the GT mask 2-channel for training
        tmp = np.zeros((240, 320, 2), dtype='uint8')
        tmp[:,:,0] = gt==255
        tmp[:,:,1] = gt==0
        gt = tmp

    return im, gt

def load_dataset(filenames, multi_channel_gt):
    X = []
    y = []
    for i, g in filenames:
        img, gt = load_images(i, g, multi_channel_gt=multi_channel_gt)
        X.append(img)
        y.append(gt)
    return np.array(X), np.array(y)

def paired_names(imgpath, gtpath, datasets, limit=0):
    result = []

    all_imgs = []
    all_gts = []

    if 'nd0405' in datasets:
        print("Finding ND0405 images...")
        imgs = glob(imgpath + '/nd_*.png')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        basenames = [n.replace('_original','') for n in basenames]
        count=0
        for bn, imname in zip(basenames, imgs):
            # infer the mask name
            fn = os.path.join(gtpath, "{}_Mask.bmp".format(bn))
            if ~os.path.exists(fn):
                # otherwise search for it
                prefix = re.match(r'^.*nd_[0-9]+d[0-9]+',fn).group(0)
                res = glob(prefix+'*')
                try:
                    fn = res[0]
                except:
                    print("Mask not found", prefix)
                    continue
            if (limit>0 and count<=limit) or limit==0:
                all_imgs.append(imname)
                all_gts.append(fn)
                count+=1
            else:
                break

    if 'biosec' in datasets:
        print("Finding BIOSEC images...")
        imgs = glob(imgpath + '/biosec_*.png')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'.png') for bn in basenames]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue

    if 'ubiris' in datasets:
        print("Finding UBIRIS images...")
        imgs = glob(imgpath + '/ubiris_*.png')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'.png') for bn in basenames]
        gtfiles = [fn.replace('ubiris_','ubiris_OperatorA_') for fn in gtfiles]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue        

    if 'casia' in datasets:
        print("Finding CASIA images...")
        imgs = glob(imgpath + '/casia_*.png')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'.png') for bn in basenames]
        gtfiles = [fn.replace('casia_','casia_OperatorA_')\
                     .replace('_original','') for fn in gtfiles]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue        

    if 'bath' in datasets:
        print("Finding BATH images...")
        imgs = glob(imgpath + '/bath_*.png')
        basenames = [os.path.basename(f)[:10] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'iris.png') for bn in basenames]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue        

    if 'warsaw-fine' in datasets:
        print("Finding WARSAW-fine images...")
        imgs = glob(imgpath + '/*.bmp')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'_Mask.bmp') for bn in basenames]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue        

    # fallback to default behavior (COMBINED DATASET)
    if len(datasets)==0:
        print("Finding ALL images...")
        imgs = glob(imgpath + '/*.bmp')
        basenames = [os.path.basename(f).split('.')[0] for f in imgs]
        gtfiles = [os.path.join(gtpath, bn+'_NIR_Mask.bmp') for bn in basenames]
        # make sure GT exists
        for im, gt in zip(imgs, gtfiles):
            if os.path.exists(gt):
                if (limit>0 and count<=limit) or limit==0:
                    all_imgs.append(im)
                    all_gts.append(gt)
                else:
                    break
            else:
                print("Mask not found", gt)
                continue        


    print("Found", len(all_imgs), "images and", len(all_gts), "masks.")
    # put all results in a list of tuples
    for im, gt in zip(all_imgs, all_gts):
        result.append((im, gt))
    return result

def IoU(gt, pred):
    intersect = np.count_nonzero(gt & pred)
    union = np.count_nonzero(gt | pred)
    return intersect/union

def pixelwise_crossentropy(target, output):
    import keras.backend as K
    output = K.tf.clip_by_value(output, K.epsilon(), 1. - K.epsilon())
    pw = - K.tf.reduce_sum(target * K.log(output))
    return pw

def tf_iou(target, output):
    import keras.backend as K
    gt = K.argmax(target)>0
    pred = K.argmax(output)>0
    intersect = K.sum(K.tf.cast(K.tf.logical_and(gt, pred), K.floatx()))
    union = K.sum(K.tf.cast(K.tf.logical_or(gt, pred), K.floatx()))
    return intersect/union

def soft_dice(target, output):
    import keras.backend as K
    axes = list(range(1, len(output.shape)-1))
    numerator = 2. * K.sum(output * target, axes)
    denominator = K.sum(K.square(output) + K.square(target), axes)
    return 1. - K.mean(numerator/(denominator+K.epsilon()))

def joint_loss(target, output):
    return pixelwise_crossentropy(target,output)+soft_dice(target,output)

def pwce_tv(target, output):
    import keras.backend as K
    tv = K.tf.reduce_sum(K.tf.image.total_variation(output))
    output = K.tf.clip_by_value(output, K.epsilon(), 1. - K.epsilon())
    pw = - K.tf.reduce_sum(target * K.log(output))
    return pw + tv
