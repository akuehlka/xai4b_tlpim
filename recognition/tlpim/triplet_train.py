import os
import argparse
import sys
import time
import cv2
import collections

from itertools import zip_longest

import numpy as np
import pandas as pd
from keras import Model
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, CSVLogger
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from keras_vggface import VGGFace
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.metrics import roc_curve, auc

from tlpim.feature_extractors.resnet import resnet_model
import tlpim.memcache as memcache

OUTPUT_LAYER_SIZE = 128
VALIDATION_SIZE = 2048

# network input image dimensions
HEIGHT = 224
WIDTH = 224

def load_file_names(path_to_dataset, exclude=[]):
    # initialize containers for data and labels
    data = []
    labels = []

    # list_images() does not follow symlinks to directories
    # imagePaths = list(paths.list_images(path_to_dataset))
    imagePaths = glob(path_to_dataset+'/**', recursive=True)
    imagePaths = [a for a in imagePaths 
        if (a.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) and 
        not any(substr in a for substr in ['highlights','wrinkles','iris']))]

    # load data from patches directory into a numpy array
    print("[INFO] describing images in the dataset")
    for i, imagePath in enumerate(imagePaths):
        # read the i-th image


        # extract the label
        path = imagePaths[i].split(os.path.sep)
        label = path[-2]

        filename = path[-1]

        if filename not in exclude:
            data.append(filename)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)

    [unique, counts] = np.unique(labels, return_counts=True)

    return data, labels, unique, counts


def prepare_data(training_folder, masks_folder, preload_train=True, ds_meta=None, pmi=0, val_set=(), subset=(), partition=True):

    def preload_helper(x_, y_):
        x_imgs_ = []
        x_masks_ = []
        x_excmasks_ = []
        for fn, label in zip(x_, y_):
            image, irismask, excmask = preload(fn, label, training_folder, masks_folder)
            x_imgs_.append(image)
            x_masks_.append(irismask)
            x_excmasks_.append(excmask)
        x_imgs_ = np.array(x_imgs_)
        x_masks_ = np.array(x_masks_)
        x_excmasks_ = np.array(x_excmasks_)
        return x_imgs_, x_masks_, x_excmasks_

    # make sure parameters do not conflict
    assert not(any((any(val_set),any(subset))) and ds_meta is None), 'Metadata file parameter required for defining subsets.'

    # load/process metadata
    excluded_files=[]
    df = pd.DataFrame()
    if ds_meta:
        df = pd.read_csv(ds_meta)
        df = df[df.wavelength=='NIR']

        tmp = df
        if subset[0]:  # use only DATASET01
            tmp = df[df.dataset=='DATASET01']
            print("Using only DATASET01 for training and testing.")

        if subset[1]:  # use only warsaw
            tmp = df[df.dataset=='Warsaw']
            print("Using only Warsaw for training and testing.")

        if pmi > 0:  # exclude also files by PMI
            tmp = tmp[tmp.pmi<=pmi]
            print(f'Excluding images of PMI higher than {pmi}.')

        excluded_files = list(set(df.filename_new.values.tolist())-set(tmp.filename_new.values.tolist()))

    print("Loading in filenames...")
    x, y, unique, counts = load_file_names(training_folder, exclude=excluded_files)

    x_train = []
    masks_train = []
    x_test = []
    masks_test = []
    y_train = []
    y_test = []

    if partition:
        print("Creating train/test splits")
        training_proportion = 0.8
        num_training = int(training_proportion * len(unique))
        np.random.seed(42)
        rand_train_indices = np.random.choice(unique, size=num_training, replace=False)

        if val_set[0]:
            print("Using DATASET01 as the testing set.")
            testlist = df[df.dataset=='DATASET01'].filename_new.values.tolist()
            trainlist = df[df.dataset!='DATASET01'].filename_new.values.tolist()
            x_test = testlist
            y_test = [sid[0:6] for sid in testlist]
            x_train = trainlist
            y_train = [sid[0:6] for sid in trainlist]
        elif val_set[1]:
            # use Warsaw as test set
            print("Using Warsaw as the testing set.")
            testlist = df[df.dataset=='Warsaw'].filename_new.values.tolist()
            trainlist = df[df.dataset!='Warsaw'].filename_new.values.tolist()
            x_test = testlist
            y_test = [sid[0:6] for sid in testlist]
            x_train = trainlist
            y_train = [sid[0:6] for sid in trainlist]
        else:
            # Generate subject disjoint train/test split
            for sub_id in unique:
                # sub_id = str(sub_id)
                indices = np.where(y == sub_id)[0]
                vals = x[indices]

                if sub_id in rand_train_indices:
                    for qwerty, im in enumerate(vals):
                        x_train.append(im)
                        y_train.append(sub_id)
                else:
                    for qwerty, im in enumerate(vals):
                        x_test.append(im)
                        y_test.append(sub_id)
    else:
        # for evaluation, no need to partition
        x_test = x
        y_test = y


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # pre-load all images in RAM
    memcache.x_train_imgs, memcache.x_train_masks, memcache.x_train_excmasks = preload_helper(x_train, y_train)
    memcache.x_test_imgs, memcache.x_test_masks, memcache.x_test_excmasks = preload_helper(x_test, y_test)

    print("Length of training subset: " + str(len(x_train)))
    print("Length of testing subset: " + str(len(x_test)))

    return x_train, y_train, x_test, y_test, x


def occlude_masks(inputs):
    target_size = 224*224
    sample_1 = inputs[0][0]
    sample_2 = inputs[0][1]
    mask = inputs[1]

    # mask out bits that are 0 from the mask
    masked_sample_1 = tf.boolean_mask(sample_1, mask)
    masked_sample_2 = tf.boolean_mask(sample_2, mask)
    # Get the mean distance difference for each value in the feature vector
    score = K.mean(K.abs(masked_sample_1 - masked_sample_2))
    print((K.mean(score)))

    return [[1, 2], score]


# ## Triplet NN
def triplet_loss(total_length, euclidean=True):
    def triplet_loss_(y_true, y_pred, alpha=0.5, epsilon=1e-8):
        """
        Implementation of the triplet loss function 
        Arguments:
        y_true -- contains the flattened and concatenated masks for anchor/positive and anchor/negative samples
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        # Extract anchor, positive and negative embeddings
        anchor = y_pred[:, 0:int(total_length * 1 / 3)]
        positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
        negative = y_pred[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]

        # Extract anchor, positive and negative masks
        m_anchor = y_true[:, 0:int(total_length * 1 / 3)]
        m_positive = y_true[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
        m_negative = y_true[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]

        # N is the # of (sigmoid) output dimensions 
        N = int(total_length/3)
        # beta should be the # of output dimensions
        beta = N

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),1)
        
        pos_dist = -tf.log(-tf.divide((pos_dist),beta)+1+epsilon)
        neg_dist = -tf.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
        
        return neg_dist + pos_dist

    return triplet_loss_


def triplet_model(shape, feat_type='resnet', base_weights=None, full_weights=None,
    train_base_model=False):

    if full_weights:
        return load_model(full_weights, compile=False)
    
    assert feat_type=='resnet' or feat_type=='densenet', "Invalid type of network for feature extraction."

    if feat_type=='resnet':
        # load the base VGGFace-ResNet model 
        # note that pre-trained network had 2000 classes
        basemodel = resnet_model(shape, 2000)
        # define layer for feature extraction (11th conv block, according to Nguyen, 2017)
        vgg_outname='conv5_1_1x1_proj'
        vgg_outlayer=basemodel.get_layer(vgg_outname).output
        # 1x1 convolution for dimensionality reduction
        reduceddims = Conv2D(512,(1,1),activation='relu')(vgg_outlayer)
        flatoutput = Flatten()(reduceddims)

    if base_weights:
        basemodel.load_weights(base_weights)
    #freeze weights

    if train_base_model:
        print(f'Updating weights in base model "{feat_type}"')
    for l in basemodel.layers:
        l.trainable=train_base_model

    dense1 = Dense(128, activation='sigmoid')(flatoutput)

    shared_backbone = Model(inputs=basemodel.input, outputs=dense1, name=f"vggface_{feat_type}_enc")

    shared_backbone.summary()

    # put together the triplet model
    anchor_input = Input(shape, name='anchor_input')
    positive_input = Input(shape, name='positive_input')
    negative_input = Input(shape, name='negative_input')

    enc_anchor = shared_backbone(anchor_input)
    enc_positive = shared_backbone(positive_input)
    enc_negative = shared_backbone(negative_input)

    merged_vector = concatenate(
        [enc_anchor, enc_positive, enc_negative],
        axis=-1,
        name='merged_layer'
    )

    model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=merged_vector, 
        name='triplet_model'
    )

    return model


def preload(filename, label, path_to_data, path_to_masks, size=(HEIGHT,WIDTH,1)):
    def load_and_resize(impath):
        try:
            tmp = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
            rsim = cv2.resize(tmp, size[0:2], interpolation=cv2.INTER_AREA).astype(np.uint8)
            return rsim
        except Exception:
            print(f"Could not load {impath}, returning empty image.")
            return np.zeros(tuple(size[0:2]), dtype=np.uint8)

    label = str(label)
    image = load_and_resize(os.path.join(path_to_data, label, filename))

    # load three individual masks
    maskname = filename.replace('.png','_iris.png')
    if 'jpg' in filename:
        maskname = filename.replace(".jpg", "_iris.png")
    elif 'bmp' in filename:
        maskname = filename.replace(".bmp", "_iris.png")
    mask_path = os.path.join(path_to_masks, label, maskname)
    iris_mask = load_and_resize(mask_path)

    maskname = filename.replace('.png','_highlights.png')
    if 'jpg' in filename:
        maskname = filename.replace(".jpg", "_highlights.png")
    elif 'bmp' in filename:
        maskname = filename.replace(".bmp", "_highlights.png")
    mask_path = os.path.join(path_to_masks, label, maskname)
    hl_mask = load_and_resize(mask_path)

    maskname = filename.replace('.png','_wrinkles.png')
    if 'jpg' in filename:
        maskname = filename.replace(".jpg", "_wrinkles.png")
    elif 'bmp' in filename:
        maskname = filename.replace(".bmp", "_wrinkles.png")
    mask_path = os.path.join(path_to_masks, label, maskname)
    wr_mask = load_and_resize(mask_path)

    # combine highlights and wrinkles in an exclusion mask
    exc_mask = np.logical_not(hl_mask | wr_mask)

    return image, iris_mask, exc_mask


def get_batch(batch_size, data, labels_encoded, unique, counts, model, path_to_data, path_to_masks, semihard=False, embed_masks=False):
    # All unique labels
    n_classes = unique
    target_size = OUTPUT_LAYER_SIZE  # number of neurons in the output layer of the model
    total_length = model.output_shape[-1]

    # initialize 2 empty arrays for the input image batch
    triplets = [[] for asdf in range(3)]

    # # initialize vector for storing the targets
    targets = np.zeros((batch_size, total_length))
    target_names = []

    bs = min(batch_size, len(n_classes))
    classes = np.random.choice(n_classes, size=(bs,), replace=False)
    i = 0
    while len(triplets[0]) < batch_size:
        if i == len(classes):
            i = 0
        # pick the first class
        class_1 = classes[i]
        # determine the number of available examples
        available_examples_1 = counts[unique == class_1]
        # if there's only one sample, we can't form a triplet
        if available_examples_1<2:
            i += 1
            continue

        # randsample two examples from this class
        idx_1 = np.random.choice(max(available_examples_1[0] - 1, 2), size=2, replace=False)

        # now get the image data from the data array
        # all samples from this class at first
        class_1_indices = np.where(labels_encoded == class_1)
        # rand_class_1 = random.randint(0, len(class_1_indices[0] - 1))

        anchor_ptr = class_1_indices[0][idx_1[0]]
        anchor_name = data[anchor_ptr]
        if embed_masks:
            anchor_mask = memcache.x_train_masks[anchor_ptr,...]
            anchor_exc_mask = memcache.x_train_excmasks[anchor_ptr,...]
            anchor = np.stack([
                    memcache.x_train_imgs[anchor_ptr,...],
                    anchor_mask,
                    anchor_exc_mask
                ], axis=2)
        else:
            anchor_mask = memcache.x_train_masks[anchor_ptr,...,np.newaxis]
            anchor_exc_mask = memcache.x_train_excmasks[anchor_ptr,...,np.newaxis]
            anchor = np.repeat(memcache.x_train_imgs[anchor_ptr,...,np.newaxis],3,axis=2)

        positive_ptr = class_1_indices[0][idx_1[1]]
        positive_name = data[positive_ptr]
        if embed_masks:
            positive_mask = memcache.x_train_masks[positive_ptr,...]
            positive_exc_mask = memcache.x_train_excmasks[positive_ptr,...]
            positive = np.stack([
                    memcache.x_train_imgs[positive_ptr,...],
                    positive_mask,
                    positive_exc_mask
                ], axis=2)
        else:
            positive_mask = memcache.x_train_masks[positive_ptr,...,np.newaxis]
            positive_exc_mask = memcache.x_train_excmasks[positive_ptr,...,np.newaxis]
            positive = np.repeat(memcache.x_train_imgs[positive_ptr,...,np.newaxis],3,axis=2)

        # Anchor sample
        triplets[0].append(anchor)

        # Positive Sample
        triplets[1].append(positive)
        target_names.append(class_1)

	    # Generate combined mask for anchor positive pair
        combined_mask_ap = (positive_exc_mask.astype(int) & anchor_exc_mask.astype(int))
        combined_mask_ap = combined_mask_ap.reshape((HEIGHT*WIDTH))

        ## BEGIN TRIPLET MINING...
        batch_samples = [np.zeros((len(classes), HEIGHT, WIDTH, 3)) for asdf in range(3)]
        batch_masks = np.zeros((len(classes), target_size,))
        for l in range(min(batch_size, len(n_classes))):
            # select a random class different from the anchor
            rand_class = np.random.choice(classes[np.arange(len(classes))!=i])
            # Indices of this second class
            indices = np.where(labels_encoded == rand_class)
            if len(indices[0])==1:
                # there's no samples to randomize
                rand = 0
            else:
                # Select random sample from this class
                rand = np.random.randint(0, high=len(indices[0]) - 1)
            rand_index = indices[0][rand]
            sample_name = data[rand_index]
            if embed_masks:
                imp_mask = memcache.x_train_masks[rand_index,...]
                exc_mask = memcache.x_train_excmasks[rand_index,...]
                sample = np.stack([
                        memcache.x_train_imgs[rand_index,...],
                        imp_mask,
                        exc_mask
                    ], axis=2)
            else:
                imp_mask = memcache.x_train_masks[rand_index,...,np.newaxis]
                exc_mask = memcache.x_train_excmasks[rand_index,...,np.newaxis]
                sample = np.repeat(memcache.x_train_imgs[rand_index,...,np.newaxis],3,axis=2)

            batch_samples[0][l, :, :, :] = anchor
            batch_samples[1][l, :, :, :] = positive
            batch_samples[2][l, :, :, :] = sample

            # Generate combined mask for anchor and potential negative
            combined_mask_an = (exc_mask.astype(int) & anchor_exc_mask.astype(int))
            combined_mask_an = combined_mask_an.reshape((HEIGHT*WIDTH))
            
        embeddings = model.predict_on_batch([batch_samples[0], batch_samples[1], batch_samples[2]])
        anc_emb = embeddings[:, 0:int(total_length * 1 / 3)][0]
        pos_emb = embeddings[:, int(total_length * 1 / 3):int(total_length * 2 / 3)][0]

        distance_ap = np.mean(np.abs(anc_emb - pos_emb))

        min_distance = float("inf")
        best_mask = np.zeros(target_size)
        num_oob = 0
        for pos, embedding in enumerate(embeddings):
            combined_mask_an = batch_masks[pos]

            neg_emb = embedding[int(total_length * 2 / 3):int(total_length * 3 / 3)].reshape(target_size)

            distance_an = np.mean(np.abs(anc_emb - neg_emb))

            if semihard:
                if (distance_an < min_distance) and \
                   (distance_ap < distance_an):
                   # select this as the best negative sample and stop mining
                   best_negative = batch_samples[2][pos,:,:,:]
                   break
            else:
                if distance_an < min_distance: # and distance_an > distance_ap:
                    # choose the minimum distance as the negative triplet (HARD)
                    min_distance = distance_an
                    best_negative = batch_samples[2][pos, :, :, :]
                    best_mask = combined_mask_an.reshape(target_size)
        try:
            triplets[2].append(best_negative)
        except:
            print("Could not find a best negative sample: Exploding gradients?")
            best_negative = batch_samples[2][pos, :, :, :]  # use the las one....

        i += 1

    triplets[0] = np.array(triplets[0])
    triplets[1] = np.array(triplets[1])
    triplets[2] = np.array(triplets[2])

    return triplets, targets, target_names


# Create a validation set, this does not do any mining just generates random triplets
def get_val(batch_size, data, labels_encoded, unique, counts, path_to_data, path_to_masks, embed_masks=False):
    classes = unique
    n_examples = counts
    target_size = HEIGHT*WIDTH

    # initialize triplet arrays
    triplets = [[] for asdf in range(3)]

    # initialize vector for storing the targets
    targets = np.zeros((batch_size, target_size*3))
    target_names = []

    # now iterate over batch and pick respective pairs
    ij = 0
    while len(triplets[0]) < batch_size:
        if ij == len(classes):
            ij = 0

        # pick the first class
        rand_class1 = np.random.randint(0, high=len(classes) - 1)
        class_1 = classes[rand_class1]
        available_examples_1 = n_examples[rand_class1]

        # if there's only one sample, we can't form a triplet
        if available_examples_1<2:
            ij += 1
            continue

        # randsample two examples from this class
        idx_1 = np.random.choice(max(available_examples_1 - 1, 2), size=2, replace=False)

        # now get the image data from the data array
        # all samples from this class at first
        class_1_indices = np.where(labels_encoded == class_1)
        # idx_1[0] = index of anchor

        anchor_ptr = class_1_indices[0][idx_1[0]]
        anchor_name = data[anchor_ptr]
        if embed_masks:
            anchor_mask = memcache.x_test_masks[anchor_ptr,...]
            anchor_exc_mask = memcache.x_test_excmasks[anchor_ptr, ...]
            anchor = np.stack([
                memcache.x_test_imgs[anchor_ptr,...],
                anchor_mask,
                anchor_exc_mask
            ], axis=2)
        else:
            anchor_mask = memcache.x_test_masks[anchor_ptr,...,np.newaxis]
            anchor_exc_mask = memcache.x_test_excmasks[anchor_ptr, ...,np.newaxis]
            anchor = np.repeat(memcache.x_test_imgs[anchor_ptr,...,np.newaxis],3,axis=2)
        
        positive_ptr = class_1_indices[0][idx_1[1]]
        positive_name = data[positive_ptr]
        if embed_masks:
            positive_mask = memcache.x_test_masks[positive_ptr,...]
            positive_exc_mask = memcache.x_test_excmasks[positive_ptr,...]
            positive = np.stack([
                memcache.x_test_imgs[positive_ptr,...],
                positive_mask,
                positive_exc_mask
            ], axis=2)
        else:
            positive_mask = memcache.x_test_masks[positive_ptr,...,np.newaxis]
            positive_exc_mask = memcache.x_test_excmasks[positive_ptr,...,np.newaxis]
            positive = np.repeat(memcache.x_test_imgs[positive_ptr,...,np.newaxis],3,axis=2)

        # Anchor sample
        triplets[0].append(anchor)

        # Positive Sample
        triplets[1].append(positive)
        target_names.append(class_1)

        combined_mask_ap = (positive_exc_mask.astype(int) & anchor_exc_mask.astype(int))

        # Select second random class, different from anchor/positive
        negative_indices = np.arange(len(classes))!=rand_class1
        rand_class2 = np.random.choice(np.where(negative_indices)[0])
        class_2 = classes[rand_class2]

        # do the same stuff as for the genuines
        available_examples_2 = n_examples[rand_class2]
        idx_2 = np.random.choice(max(available_examples_2 - 1,1), size=1, replace=False)
        class_2_indices = np.where(labels_encoded == class_2)
        # idx_1[0] = index of anchor
        assert class_2 != class_1, "Negative class is the same!"

        negative_ptr = class_2_indices[0][idx_2[0]]
        negative_name = data[negative_ptr]
        if embed_masks:
            negative_mask = memcache.x_test_masks[negative_ptr,...]
            negative_exc_mask = memcache.x_test_excmasks[negative_ptr,...]
            negative = np.stack([
                memcache.x_test_imgs[negative_ptr,...],
                negative_mask,
                negative_exc_mask
            ], axis=2)
        else:
            negative_mask = memcache.x_test_masks[negative_ptr,...,np.newaxis]
            negative_exc_mask = memcache.x_test_excmasks[negative_ptr,...,np.newaxis]
            negative = np.repeat(memcache.x_test_imgs[negative_ptr,...,np.newaxis],3,axis=2)

        # Negative Sample
        triplets[2].append(negative)

        combined_mask_an = (negative_exc_mask.astype(int) & anchor_exc_mask.astype(int))

        targets[ij, :] = np.concatenate((
            combined_mask_ap.reshape(target_size), 
            combined_mask_an.reshape(target_size), 
            np.zeros(target_size)
        ))

        ij += 1

    triplets[0] = np.array(triplets[0])
    triplets[1] = np.array(triplets[1])
    triplets[2] = np.array(triplets[2])

    return triplets, targets, target_names


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def calc_matching_accuracy(model, batch_size, input_images, input_labels):
    tlength=model.output_shape[-1]
    INPUT_SIZE=input_images[0].shape[0]
    Y = input_labels

    anchor_embedding_length = int(tlength * 1 / 3)
    preds = np.zeros((INPUT_SIZE, anchor_embedding_length), np.float)
    for i_range in grouper(range(INPUT_SIZE), batch_size):
        i_chunk = [i for i in i_range if i is not None]
        tmp = model.predict_on_batch([
            input_images[0][i_chunk,...], 
            input_images[1][i_chunk,...], 
            input_images[2][i_chunk,...]])

        # extract the anchor embeddings    
        preds[i_chunk,...] = tmp[:, 0:anchor_embedding_length]

    # calculates all-vs-all distances
    distances = []
    for refix, refname in enumerate(Y):
        for probeix, probename in enumerate(Y):
            if probeix<=refix:
                continue
            else:
                # euclidean distance
                dist = np.linalg.norm(abs(preds[refix] - preds[probeix]))
                distances.append((refname, probename, dist))
                # print(refname, probename, dist)

    df = pd.DataFrame(data=distances, columns=['reference','probe','distance'])
    # scores as positive probability estimates
    df['proba'] = 1-(df.distance/df.distance.max())
    df['isgenuine'] = (df.reference==df.probe).astype(int)

    gend = df[df.isgenuine==1].distance.mean()
    impd = df[df.isgenuine==0].distance.mean()
    print('Average genuine distance:', gend, 'impostor distance:', impd, 'Distance between averages:', abs(gend-impd))

    fpr, tpr, thr = roc_curve(
        df.isgenuine,
        df.proba
    )
    roc_auc = auc(fpr, tpr)
    print('Validation AUC:',roc_auc)

    tpr = len(df[(df.isgenuine==1) & (df.proba>0.5)])/len(df[df.isgenuine==1])
    tnr = len(df[(df.isgenuine==0) & (df.proba<=0.5)])/len(df[df.isgenuine==0])
    print('TPR/TNR @ thr 0.5:',tpr, '/', tnr)


def calc_validation_acc(model, batch_size, inputs_test, targets_test):
    # and evaluate the testing accuracy while we are here
    before = time.time()
    _loss_test = []
    for i_range in grouper(range(VALIDATION_SIZE), batch_size):
        i_chunk = [i for i in i_range if i is not None]
        _loss_test = model.test_on_batch([
            inputs_test[0][i_chunk,...], 
            inputs_test[1][i_chunk,...], 
            inputs_test[2][i_chunk,...]], 
            targets_test[i_chunk,...])[0]
    loss_test = np.array(_loss_test).mean()
    time_taken = time.time() - before

    print("Loss on the testing set is:" + str(loss_test) + " and ran in: " + str(time_taken))

    return np.round(loss_test, 4)


def write_log(callback, names, logs, batch_no):
    for name, value in zip (names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def write_triplet(anchor, pos, neg, label, output):
    from matplotlib import pyplot as plt

    for i in np.arange(anchor.shape[0]):
        fig = plt.figure(figsize=(20,7))
        ax = plt.subplot(1,3,1)
        ax.imshow(anchor[i,...])
        ax.set_ylabel(label[i])
        ax = plt.subplot(1,3,2)
        ax.imshow(pos[i,...])
        ax = plt.subplot(1,3,3)
        ax.imshow(neg[i,...])
        plt.savefig(os.path.join(output,f'triplet_{i}.jpg'))


def main(args):
    # Set up hardware
    if args.gpu is not None:
        GPU_NO = ','.join([str(n) for n in args.gpu])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.visible_device_list = GPU_NO
        set_session(tf.Session(config=config))

    # Get output files figured out
    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    if args.pmi_limit>0:
        assert args.ds_meta is not None, "Dataset metadata must be provided for PMI limiting."
    if args.val_dataset01:
        assert args.ds_meta is not None, "Dataset metadata must be provided to use DATASET01 as validation set."
    if args.val_warsaw:
        assert args.ds_meta is not None, "Dataset metadata must be provided to use WARSAW as validation set."

    # Get data ready
    x_train, y_train, x_test, y_test, _ = prepare_data(args.training_folder, args.masks_folder, 
        ds_meta=args.ds_meta, pmi=args.pmi_limit, 
        val_set=(args.val_dataset01, args.val_warsaw,),
        subset=(args.dataset01_only, args.warsaw_only))
    [unique, counts] = np.unique(y_train, return_counts=True)
    [unique_test, counts_test] = np.unique(y_test, return_counts=True)

    # save a list of test samples 
    pd.DataFrame({'filename':x_test}).to_csv(f"{args.output}/testset.txt", index=False)
    # save a list of train samples 
    pd.DataFrame({'filename':x_train}).to_csv(f"{args.output}/trainset.txt", index=False)
    if args.only_gen_partition:
        print("Data partition written.")
        sys.exit(0)

    if args.solver == 'adam':
        adam = Adam(lr=args.lr, clipnorm=1.0, clipvalue=0.5)
    elif args.solver == 'sgd':
        adam = SGD(lr=args.lr)
    else:
        raise Exception('Solver not recognized!')

    data_shape = (HEIGHT, WIDTH, 3)
    n_subjects = len(unique)

    feat_type = args.type
    base_weights = None
    if args.vggfaces_weights:
        base_weights = args.densenet_weights

    print("Creating model")
    if args.gpu is None or len(args.gpu)==1:
        model = triplet_model(data_shape, base_weights=base_weights, feat_type=feat_type, 
            train_base_model=args.update_base_model)
    else:
        # multi-gpu
        with tf.device('/cpu:0'):
            model = triplet_model(data_shape, base_weights=base_weights, feat_type=feat_type, 
                train_base_model=args.update_base_model)
        # create a parallel model
        model = multi_gpu_model(model, gpus=len(args.gpu))
    model.summary()

    # Callbacks
    tensorboard = TensorBoard(log_dir=os.path.join(output,'logs'),)

    reduceLR = ReduceLROnPlateau(monitor='loss',
                                 patience=2,
                                 epsilon=0.1,
                                 verbose=1,
                                 mode='min')

    file_name = args.prefix + '_.h5py'
    path_to_save = os.path.join(output, file_name)

    model_saver = ModelCheckpoint(path_to_save,
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='min', period=1)

    early_stop = EarlyStopping(monitor='acc',
                               min_delta=0.001,
                               patience=5,
                               verbose=1,
                               mode='max')

    csv_logger = CSVLogger('training_natural.log')

    callbacks = [model_saver, reduceLR, early_stop, tensorboard, csv_logger]

    tlength = model.output_shape[-1]
    model.compile(
        optimizer=adam,
        loss=triplet_loss(
            total_length=tlength,
            euclidean=(not args.cosine_dist)
        ),
        metrics=['acc']
    )

    tensorboard.set_model(model)
    train_names = ['train_loss']
    val_names = ['val_loss']

    # create a fixed batch of testing data
    print("[INFO] Generating validation set...")
    val_before = time.time()
    global VALIDATION_SIZE 
    VALIDATION_SIZE = len(y_test)
    inputs_test, targets_test, target_test_names = get_val(
        VALIDATION_SIZE, x_test, y_test, unique_test, counts_test,
        args.training_folder, args.masks_folder,
        embed_masks=args.embed_masks
    )
    val_creation = time.time() - val_before
    x_test = []
    print("Validation creation time: " + str(val_creation))
    print("[INFO] Training begins...")

    batch_size = args.batch_size
    # Max training time
    n_iterations = args.n_iterations
    target_size = 256*256

    # variable to store accuracies
    accuracies = []
    accuracies_test = []

    loss_list = []
    previous_losses = collections.deque(maxlen=args.patience)
    for j in range(args.patience):
        previous_losses.append(float('inf'))

    for i in range(1, n_iterations):
        before_batch = time.time()
        inputs, targets, target_names = get_batch(
            batch_size, x_train, y_train, unique, counts, model, 
            args.training_folder, args.masks_folder,
            semihard=i<args.start_hard,
            embed_masks=args.embed_masks
        )

        loss = model.train_on_batch([inputs[0], inputs[1], inputs[2]], targets)
        write_log(tensorboard, train_names, loss, i)

        loss_list.append(loss)
        batch_time = time.time() - before_batch

        print(f"Iteration {i:d} loss: {loss[0]:e}, time: {batch_time:f}")

        if i % args.val_period == 0:
            print(("[INFO] Validating and saving model: {}/{}".format(i, n_iterations)))
            mean_loss = calc_validation_acc(model, batch_size, inputs_test, targets_test)
            print(mean_loss, np.array(previous_losses))

            if args.valauc:
                calc_matching_accuracy(model, batch_size, inputs_test, y_test)

            write_log(tensorboard, val_names, loss, i)

            loss_list = []
            if mean_loss < previous_losses[-1]:
                model.save(os.path.join(
                    args.output, 
                    args.prefix + str(i) + ".h5py"
                ))
            elif all(mean_loss >= np.array(previous_losses)) and \
                (np.std(previous_losses) < 1) and \
                i > args.min_iterations:
                print("Training complete, loss did not decrease over {} validation steps.".format(len(previous_losses)))
                break
            previous_losses.append(mean_loss)


    print("[INFO] Saving the model weights...")
    model.save(os.path.join(
        args.output, 
        args.prefix + "_final.h5py"
    ))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', '-o', required=True, help='Output folder.')
    ap.add_argument('--prefix', '-p', required=True, help='Prefix for the file.')
    ap.add_argument('--gpu', type=int, nargs='*')
    ap.add_argument('--lr', action='store', dest='lr',
                    default=0.01, type=float)
    ap.add_argument('--batch_size', action='store', type=int, default=64)
    ap.add_argument('--n_iterations', type=int, default=200000)
    ap.add_argument('--solver', action='store',
                    dest='solver', default='sgd')

    ap.add_argument('--val_period', '-vp', type=int, default=100)

    ap.add_argument('--vggfaces_weights','-w', type=str,
                    help='Pre-trained weights for VGGFace-ResNet feature extraction.')
    ap.add_argument('--densenet_weights','-dw', type=str,
                    help='Pre-trained weights for DenseNet feature extraction.')
    ap.add_argument('--training_folder', '-tf', required=True, type=str,
                    help='Path to the folder with training images.')
    ap.add_argument('--masks_folder', '-mf', required=False, type=str,
                    help='Path to the folder with highlights/wrinkles masks.')

    ap.add_argument('--start_hard','-sh',type=int, default=0,
                    help='Iteration when to start hard triplet mining.')

    ap.add_argument('--ds_meta', type=str, help='Dataset metadata (required for PMI limit)')
    ap.add_argument('--pmi_limit', type=int, default=0,
                    help='Maximim PMI value for selecting samples.')

    ap.add_argument('--val-on-dataset01',dest='val_dataset01',action='store_true')
    ap.add_argument('--val-on-warsaw',dest='val_warsaw',action='store_true')
    ap.add_argument('--dataset01-only',dest='dataset01_only',action='store_true')
    ap.add_argument('--warsaw-only',dest='warsaw_only',action='store_true')
    ap.add_argument('--valauc',dest='valauc',action='store_true')
    ap.set_defaults(val_dataset01=False, val_warsaw=False, valauc=False, dataset01_only=False, warsaw_only=False)

    ap.add_argument('--partition_only',dest='only_gen_partition', action='store_true')
    ap.set_defaults(only_gen_partition=False)

    ap.add_argument('--cosine-dist',dest='cosine_dist',action='store_true',help='Use cosine distance in loss computation.')
    ap.set_defaults(cosine_dist=False)

    ap.add_argument('--type', choices=['resnet','densenet'], default='resnet')

    ap.add_argument('--min_iterations',type=int,default=1000,help='Minimum # of iterations to train (ignores loss).')
    ap.add_argument('--patience', type=int, default=5, help='Number of iterations without loss decrease before stopping.')

    ap.add_argument('--update_base_model',dest='update_base_model',action='store_true')
    ap.add_argument('--embed_masks',dest='embed_masks',action='store_true')
    ap.set_defaults(update_base_model=False, embed_masks=False)

    args = ap.parse_args()

    main(args)
