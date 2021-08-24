import tensorflow as tf
import numpy as np
import os
import os.path as osp
import csv
import argparse
import cv2

from tlpim.triplet_train import triplet_model, prepare_data, grouper, triplet_loss
from tlpim.cam import CAMHelper
import tlpim.memcache as memcache

from scipy.spatial.distance import cosine, pdist
import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import h5py

def get_val(batch_size, data, labels_encoded, unique, counts, path_to_data, path_to_masks):
    """
    Loads an image into a triplet, 
    """
    height = 224
    width = 224
    target_size = height*width

    # since we're not training, train/test sets do not matter, so we concatenate everything
    if memcache.x_train_imgs.shape!=(0,):
        x_images = np.concatenate((memcache.x_train_imgs, memcache.x_test_imgs), axis=0)
    else:
        x_images = memcache.x_test_imgs

    batch_size_ = min(batch_size, x_images.shape[0])

    # initialize triplet arrays
    triplets = [np.zeros((batch_size_, height, width, 3)) for asdf in range(3)]

    # initialize vector for storing the targets
    targets = np.zeros((batch_size_, target_size*3))

    for i in range(batch_size_):
        input_image_3c = np.repeat(x_images[i,...,np.newaxis],3,axis=2)
        triplets[0][i,...] = input_image_3c
        triplets[1][i,...] = np.zeros_like(input_image_3c)
        triplets[2][i,...] = np.zeros_like(input_image_3c)

    return triplets, targets


def init_model(ftype,weights):
    data_shape = (224, 224, 3)

    model = triplet_model(data_shape, feat_type=ftype,full_weights=weights)
    tlength=model.output_shape[-1]
    model.compile(optimizer='adam',
                loss=triplet_loss(total_length=tlength),
                metrics=['acc'])
    return model


def predict(args):
    print('Predicting distances...')

    flags = tf.app.flags
    flags.DEFINE_string('f', '', 'kernel')
    FLAGS = flags.FLAGS

    batch_size=192

    model = init_model(args.ftype, args.weights)
    tlength=model.output_shape[-1]

    assert (args.dataset01_only or args.warsaw_only) or args.ds_meta is None, "Metadata parameter must be informed to filter between Dataset01/Warsaw."

    # Get data ready
    x_train, y_train, x_test, y_test, filesfound = prepare_data(args.image_folder, 
        args.masks_folder, preload_train=True,
        ds_meta=args.ds_meta,
        val_set=(False, False),
        subset=(args.dataset01_only, args.warsaw_only),
        partition=False)

    [unique, counts] = np.unique(y_train, return_counts=True)
    [unique_test, counts_test] = np.unique(y_test, return_counts=True)

    assert (args.testfiles is not None) or \
        (args.pairlist is not None), \
        "Either list of samples (all-vs-all) or list of pairs to be tested must be passed."

    if args.testfiles:
        # All samples to be tested are passed in a csv file
        filenames = pd.read_csv(args.testfiles).filename.tolist()
        filenames = [a for a in filenames if a not in set(filenames)-set(filesfound)]
    elif args.pairlist:
        # all samples must be extracted from the list of pairs
        pairs = pd.read_csv(args.pairlist, 
                            usecols=['file1','file2'],
                            sep=' ')
        filenames = np.unique(pairs.file1.append(pairs.file2).values)
    missingfiles = set(filenames)-set(filesfound)
    print(f"{len(missingfiles)} Files not found: {missingfiles}")

    # load ALL files found, regardless if they're to be tested or not
    INPUT_SIZE=len(filesfound)
    not_used = None
    input_images, _ = get_val(
        INPUT_SIZE, filesfound, not_used, not_used, not_used,
        args.image_folder, args.masks_folder
    )

    anchor_embedding_length = int(tlength * 1 / 3)
    preds = np.zeros((INPUT_SIZE, anchor_embedding_length), np.float)
    batch_size = min(batch_size, INPUT_SIZE)
    try:
        for i_range in grouper(range(INPUT_SIZE), batch_size):
            i_chunk = [i for i in i_range if i is not None]
            tmp = model.predict_on_batch([
                input_images[0][i_chunk,...], 
                input_images[1][i_chunk,...], 
                input_images[2][i_chunk,...]])

            # extract the anchor embeddings    
            preds[i_chunk,...] = tmp[:, 0:anchor_embedding_length]
    except Exception as e:
        raise e

    # save the embeddings
    if args.saveembeddings:
        h5f = h5py.File(f'{args.outpath}/embeddings.h5','w')
        h5f.create_dataset('embeddings', preds.shape, data=preds)
        h5f.create_dataset(
            'labels', 
            data=[x.encode('ascii','ignore') for x in filesfound],
            dtype='S20'
        )
        h5f.close()

    prefix = 'output'
    if args.testfiles:
        prefix = os.path.basename(args.testfiles).split('.')[0]
        # calculates all-vs-all distances
        distances = []
        
        if args.mahalanobis:
            print("Calculating mahalanobis distance")
            # load the covariance matrix for trainset embeddings
            covfile = os.path.join(os.path.dirname(args.weights), 'trainset_embeddings_covariance.npz')
            npzfile = np.load(covfile)
            covariance = npzfile["covariance"]
        
        if args.chebyshev:
            print("Calculating chebyshev distance")

        # restrict the evaluation to the list of files to be tested
        files_to_test = [a for a in filesfound if a in filenames]

        for refix, refname in enumerate(files_to_test):
            for probeix, probename in enumerate(files_to_test):
                if probeix>refix:
                    if args.mahalanobis:
                        dist = pdist(np.stack((preds[refix,...], preds[probeix,...]), axis=0),'mahalanobis', VI=covariance)[0]
                    elif args.chebyshev:
                        dist = pdist(np.stack((preds[refix,...], preds[probeix,...]), axis=0),'chebyshev')[0]
                    else:
                        dist = pdist(np.stack((preds[refix,...], preds[probeix,...]), axis=0),'euclidean')[0]
                    # print(refname, probename, dist)
                    distances.append((refname, probename, dist))


    elif args.pairlist:
        prefix = os.path.basename(args.pairlist).split('.')[0]
        # create a map with the embedding for each sample
        embeddings_map = {}
        for i, key in enumerate(filenames):
            embeddings_map[key] = preds[i,...]
        # loop through the pairs calculating the score
        distances = []
        for refname, probename in zip(pairs.file1.values, pairs.file2.values):
            dist = np.linalg.norm(abs(embeddings_map[refname] - embeddings_map[probename]))
            distances.append((refname, probename, dist))

    os.makedirs(args.outpath, exist_ok=True)
    fname = f'{args.outpath}/{prefix}-distances.csv'
    # update the argument
    args.scorefile = fname
    with open(fname,'w') as f:
        dw = csv.DictWriter(f, ['reference','probe','distance'])
        dw.writeheader()
        for pair in distances:
            dw.writerow({
                'reference': pair[0],
                'probe': pair[1],
                'distance': pair[2],
            })

    print('Done predicting distances.')

    print('Generating visualizations...')
    # generate heatmaps
    CAMHelper.process(
        filesfound, 
        weights=args.weights,
        images_dir=args.image_folder,
        output_dir=args.outpath
    )
    for f in filesfound:
        print(f)
        viz = make_vizs(f, args)
        cv2.imwrite(os.path.join(args.outpath,f.replace('.bmp','_viz.png')),viz)
    print('Done')



def make_vizs(imname, args):
    imname_root, imname_type = osp.splitext(imname)
    img_name = os.path.join(args.image_folder, imname_root[0:6], imname_root+".bmp")
    cropped_img = cv2.imread(img_name,0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped_img = clahe.apply(cropped_img)

    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    mask_i_name = os.path.join(args.image_folder, imname_root[0:6],imname_root+'_iris.png')
    cropped_msk_iris = cv2.imread(mask_i_name,0)
    mask_h_name = os.path.join(args.image_folder, imname_root[0:6],imname_root+'_highlights.png')
    cropped_msk_hl = cv2.imread(mask_h_name,0)
    mask_w_name = os.path.join(args.image_folder, imname_root[0:6], imname_root+'_wrinkles.png')
    cropped_msk_wr = cv2.imread(mask_w_name,0)
    hm_name = os.path.join(args.outpath, imname_root+'_heatmap.png')
    heatmap = cv2.imread(hm_name,0)
    heatmap = cv2.resize(heatmap, cropped_img.shape[:2])
    
    cm = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    output = cv2.addWeighted(cropped_img, 0.5, cm, 0.5, 0.0)

    # find iris contours and draw them
    _, mi_contours, _ = cv2.findContours(cropped_msk_iris, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(output, mi_contours, -1, (255,0,0), 3)

    # find highlights contours and draw them
    _, hl_contours, _ = cv2.findContours(cropped_msk_hl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(output, hl_contours, -1, (0,255,255), 3)

    # find wrinkles contours and draw them
    _, wr_contours, _ = cv2.findContours(cropped_msk_wr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(output, wr_contours, -1, (0,255,0), 3)

    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('action', default='all', choices=['predict'])
    ap.add_argument('--outpath',type=str, default=os.getcwd(), help='Path for output files.')
    ap.add_argument('--scorefile',type=str, help='File with scores.')
    ap.add_argument('--image_folder', type=str, 
    help="Path to training/prediction images. Images must be organized in subfolders by subject.")
    ap.add_argument('--masks_folder', type=str, help="Path to mask images.")
    ap.add_argument('--weights', type=str, help='Weights file for the network.')
    ap.add_argument('--ftype', type=str, default='resnet', choices=['resnet','densenet'], help='Type of network for feature extraction')

    ap.add_argument('--testfiles', type=str, help='CSV file with the samples to be tested.')

    ap.add_argument('--dataset01-only',dest='dataset01_only',action='store_true')
    ap.add_argument('--warsaw-only',dest='warsaw_only',action='store_true')
    ap.set_defaults(val_warsaw=False, valauc=False, dataset01_only=False, warsaw_only=False)

    ap.add_argument('--ds_meta', type=str, help='Dataset metadata (required for PMI limit)')
    ap.add_argument('--pmi_limit', type=int, default=0,
                    help='Maximim PMI value for selecting samples.')

    ap.add_argument('--pairlist', type=str, help='File with the list of pairs to be matched.')

    ap.add_argument('--mahalanobis', dest='mahalanobis', action='store_true')
    ap.add_argument('--chebyshev', dest='chebyshev', action='store_true')
    ap.set_defaults(mahalanobis=False, chebyshev=False)

    ap.add_argument('--saveembeddings', dest='saveembeddings', action='store_true', help='Save the network predictions')
    ap.set_defaults(saveembeddings=False)

    args = ap.parse_args()

    if args.action=='predict' or args.action=='all':
        assert args.image_folder is not None, "Must provide --image_folder argument."
        assert args.weights is not None, "Must provide --weights argument."
        if args.masks_folder is None:
            args.masks_folder = args.image_folder

        predict(args)


