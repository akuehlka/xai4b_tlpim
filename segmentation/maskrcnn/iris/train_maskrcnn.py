#!/usr/bin/env python
# coding: utf-8
from maskrcnn.iris import *

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
import re
import argparse
import shutil
import tempfile
import sys
from glob import glob

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

from maskrcnn.iris.util import IoU
from maskrcnn.iris.util import paired_names, load_dataset

# use a temp dir to allow multiple simultaneous jobs
OUTDIR = tempfile.TemporaryDirectory()

def get_dicts(imgnames, msknames):
    from pycocotools.mask import encode
    dataset_dicts = []
    for i, img, msk in [(a[0], a[1][0], a[1][1]) for a in enumerate(zip(imgnames, msknames))]:
        record = {}
        height, width = cv2.imread(img).shape[:2]
        mskimg = cv2.imread(msk,0)
        mskbool = (mskimg==255).astype(np.uint8)

        record["file_name"] = img
        record["image_id"] = i
        record["height"] = height
        record["width"] = width
        
        # find a bounding box for the mask
        _, contours, _ = cv2.findContours(mskimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0

        bbox = [None, None, None, None]
        segm = {}
        if contours:
            cnt = contours[0]
            largest_contour = cnt
            for c in contours:
                area = cv2.contourArea(c)
                if (area > largest_area):
                    largest_area = area
                    largest_contour = c
            rect = cv2.boundingRect(largest_contour)
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            segm = encode(np.asarray(mskbool, order='F'))
        
        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": segm,
            "category_id": 0,
        }
        record["annotations"] = [obj]
        
        if not any(bbox):
            # discard warsaw images with empty annotations
            print("Empty annotation >>", record['file_name'])
        else:
            dataset_dicts.append(record)
    return dataset_dicts


def init_detectron(tset='live', weightsfile=None, lr=0.00025, mi=5000, numclasses=1, 
    live_impath='./samples-live',
    live_gtpath='./samples-live',
    dataset01_impath='./samples',
    dataset01_gtpath='./samples',
    warsaw_impath=None,
    warsaw_gtpath=None
):
    maskrcnn = MaskRCNN()

    IMGPATH=live_impath
    MSKPATH=live_gtpath
    print("Loading live images from ",IMGPATH, MSKPATH)
    # HERE WE SPECIFY ONLY THE LIVE DATASETS FOR TRAINING:
    livefiles = paired_names(IMGPATH, MSKPATH, ['nd0405','casia','bath','biosec','ubiris'])

    IMGPATH=dataset01_impath
    MSKPATH=dataset01_gtpath
    print("Loading DATASET01 images from ",IMGPATH, MSKPATH)
    dataset01_files = paired_names(IMGPATH, MSKPATH, [])
    maskrcnn.DATASET01_IMGNAMES = [r[0] for r in dataset01_files]
    maskrcnn.DATASET01_MSKNAMES = [r[1] for r in dataset01_files]

    if warsaw_impath is None or warsaw_gtpath is None:
        # find images/masks directly in the warsaw dataset
        IMGPATH='./samples'
        MSKPATH='./samples'
        print("Loading warsaw images from ",IMGPATH, MSKPATH)
        warsaw_files = paired_names(IMGPATH, MSKPATH, [])
    else:
        # find images/masks distributed with the live training set
        IMGPATH=warsaw_impath
        MSKPATH=warsaw_gtpath
        print("Loading warsaw-fine images from ",IMGPATH, MSKPATH)
        warsaw_files = paired_names(IMGPATH, MSKPATH, ['warsaw-fine'])
    maskrcnn.WARSAW_IMGNAMES = [r[0] for r in warsaw_files]
    maskrcnn.WARSAW_MSKNAMES = [r[1] for r in warsaw_files]

    trainfiles = livefiles
    if tset == 'live+warsaw':
        trainfiles = np.vstack((livefiles, warsaw_files))
    elif tset == 'warsaw':
        trainfiles = warsaw_files
    
    maskrcnn.TRAIN_IMGNAMES = [r[0] for r in trainfiles]
    maskrcnn.TRAIN_MSKNAMES = [r[1] for r in trainfiles]

    # LOAD DATASET DICTIONARIES
    maskrcnn.trainds_dicts = get_dicts(maskrcnn.TRAIN_IMGNAMES, maskrcnn.TRAIN_MSKNAMES)
    print('Loaded TRAIN dataset metadata.')
    maskrcnn.dataset01_dicts = get_dicts(maskrcnn.DATASET01_IMGNAMES, maskrcnn.DATASET01_MSKNAMES)
    maskrcnn.warsaw_dicts = get_dicts(maskrcnn.WARSAW_IMGNAMES, maskrcnn.WARSAW_MSKNAMES)
    print('Loaded TEST datasets metadata.')

    # REGISTER DATASETS
    DatasetCatalog.register("wwlive_train", lambda i=maskrcnn.TRAIN_IMGNAMES,m=maskrcnn.TRAIN_MSKNAMES: get_dicts(i, m))
    MetadataCatalog.get('wwlive_train').set(stuff_classes=["iris"])
    maskrcnn.iris_metadata = MetadataCatalog.get("wwlive_train")

    DatasetCatalog.register("dataset01_test", lambda i=maskrcnn.DATASET01_IMGNAMES,m=maskrcnn.DATASET01_IMGNAMES: get_dicts(i, m))
    MetadataCatalog.get('dataset01_test').set(stuff_classes=["iris"])
    maskrcnn.dataset01_metadata = MetadataCatalog.get("dataset01_test")

    DatasetCatalog.register("wwpostm_test", lambda i=maskrcnn.WARSAW_IMGNAMES,m=maskrcnn.WARSAW_IMGNAMES: get_dicts(i, m))
    MetadataCatalog.get('wwpostm_test').set(stuff_classes=["iris"])
    maskrcnn.wwpostm_metadata = MetadataCatalog.get("wwpostm_test")

    # SET UP MASK-RCNN
    maskrcnn.cfg = get_cfg()
    maskrcnn.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    maskrcnn.cfg.DATASETS.TRAIN = ("wwlive_train",)
    maskrcnn.cfg.DATASETS.TEST = ()
    maskrcnn.cfg.DATALOADER.NUM_WORKERS = 2
    maskrcnn.cfg.INPUT.MASK_FORMAT = 'bitmask'
    
    maskrcnn.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    if weightsfile:
        print("Using initial weights from ", weightsfile)
        maskrcnn.cfg.MODEL.WEIGHTS = weightsfile

    maskrcnn.cfg.SOLVER.IMS_PER_BATCH = 8
    maskrcnn.cfg.SOLVER.BASE_LR = lr  
    maskrcnn.cfg.SOLVER.MAX_ITER = mi
    maskrcnn.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    maskrcnn.cfg.MODEL.ROI_HEADS.NUM_CLASSES = numclasses
    maskrcnn.cfg.OUTPUT_DIR = OUTDIR.name

    return maskrcnn


def train(maskrcnn, savemodel):
    # TRAINING
    trainer = DefaultTrainer(maskrcnn.cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    if savemodel:
        shutil.copy(
            OUTDIR.name + '/model_final.pth',
            savemodel
        )


def evaluate(maskrcnn):
    # LOAD MODEL FROM PERMANENT STORAGE
    maskrcnn.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

    # EVALUATE ON DATASET01
    maskrcnn.cfg.DATASETS.TEST = ("dataset01_test", )
    predictor = DefaultPredictor(maskrcnn.cfg)

    # to search for the mask files
    dict_masks = {re.findall(r'[0-9]{4}_[LR]_[0-9]+_[0-9]', x)[0]: x for x in maskrcnn.DATASET01_MSKNAMES}

    results = []
    for d in maskrcnn.dataset01_dicts:
        img = cv2.imread(d['file_name'])
        mskkey = os.path.basename(d['file_name']).split('.')[0]
        msk = cv2.imread(dict_masks[mskkey],0)>0
        
        outputs = predictor(img)
        pred = outputs["instances"].pred_masks.cpu().numpy()[0,...]
        
        results.append(IoU(msk, pred))
    print(np.mean(results))


    # EVALUATE ON WARSAW
    maskrcnn.cfg.DATASETS.TEST = ("warsaw_test", )
    predictor = DefaultPredictor(maskrcnn.cfg)

    # search for the mask files
    dict_wwpostm = {re.findall(r'[0-9]{4}_[LR]_[0-9]+_[0-9]', x)[0]: x for x in maskrcnn.WARSAW_MSKNAMES}

    results = []
    for d in maskrcnn.warsaw_dicts:
        img = cv2.imread(d['file_name'])
        mskkey = os.path.basename(d['file_name']).split('.')[0]
        if mskkey not in dict_wwpostm.keys():
            print(mskkey, 'not found')
            continue
        msk = cv2.imread(dict_wwpostm[mskkey],0)>0
        
        outputs = predictor(img)
        if outputs['instances'].pred_masks.cpu().numpy().size > 0:
            pred = outputs["instances"].pred_masks.cpu().numpy()[0,...]

            results.append(IoU(msk, pred))
        else:
            print(mskkey, 'no output')
    print(np.mean(results))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', dest='run_training', action='store_true')
    ap.set_defaults(run_training=False)
    ap.add_argument('--tset_pm', dest='tset', type=str, default='live', 
        help='If present, this argument includes warsaw post-mortem images in the training. Default is Warsaw-Coarse.')
    ap.add_argument('--live_impath',type=str, help='Override path for live training images.')
    ap.add_argument('--live_gtpath',type=str, help='Override path for live training ground truth masks.')
    ap.add_argument('--dataset01_impath', type=str, help='Override path for DATASET01 images.')
    ap.add_argument('--dataset01_gtpath', type=str, help='Override path for DATASET01 ground truth masks.')
    ap.add_argument('--warsaw_impath', type=str, help='Override path for Warsaw post-mortem images. Default is Coarse')
    ap.add_argument('--warsaw_gtpath', type=str, help='Override path for Warsaw post-mortem ground truth masks.')
    ap.add_argument('--weights', type=str, help='Specify the file from which initial weights should be loaded. Default is ResNet(ImageNet).')
    ap.add_argument('--eval', dest='run_eval', action='store_true')
    ap.set_defaults(run_eval=False)
    ap.add_argument('--lr', type=float, default=0.00025, help='Learning rate.')
    ap.add_argument('--mi', type=int, default=5000, help='Max iterations.')
    ap.add_argument('--savemodel', type=str, help='Path to file where to save the newly trained model.')
    ap.add_argument('--outputclasses', type=int, default=1, help='Number of output classes (determines the shape of the network)')
    args = ap.parse_args()

    initargs = {k:v for k, v in vars(args).items() if (any(x in k for x in ['_impath', '_gtpath'])) and v is not None}
    maskrcnn = init_detectron(
        tset=args.tset,
        weightsfile=None if not args.weights else args.weights,
        lr=args.lr,
        mi=args.mi, 
        numclasses=args.outputclasses,
        **initargs
    )
    if args.run_training:
        train(maskrcnn, savemodel='' if not args.savemodel else args.savemodel)
    if args.run_eval:
        evaluate(maskrcnn)

