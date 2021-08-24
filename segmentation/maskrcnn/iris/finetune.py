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
import os
from glob import glob
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer

# use a temp dir to allow multiple simultaneous jobs
OUTDIR = tempfile.TemporaryDirectory()

def get_dicts(imgnames, msknames, hlnames, wrnames, exclude_no_gt=True):
    from pycocotools.mask import encode
    dataset_dicts = []
    for i, img, msk, hl, wr in [(a[0], a[1][0], a[1][1], a[1][2], a[1][3]) for a in enumerate(zip(imgnames, msknames, hlnames, wrnames))]:
        record = {}
        if not os.path.exists(img):
            print(f'File not found: {img}')
            continue
        height, width = cv2.imread(img).shape[:2]

        # just in case we don't have a GT image
        mskimg = np.zeros((width, height)).astype(np.uint8)
        mkkbool = mskimg.copy()
        if msk:
            mskimg = cv2.imread(msk,0)
            mskbool = (mskimg==255).astype(np.uint8)

        hlimg = None
        if len(hl)>0:
            hlimg = cv2.imread(hl, 0)

        wrimg = None
        if len(wr)>0:
            wrimg = cv2.imread(wr, 0)

        record["file_name"] = img
        record["image_id"] = i
        record["height"] = height
        record["width"] = width
        
        # find a (single) bounding box for the iris mask
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
            "category_id": 0, # iris
        }
        record["annotations"] = [obj]

        if exclude_no_gt and (not any(bbox)):
            # discard warsaw images with empty annotations
            print("Empty annotation >>", record['file_name'])
            continue

        # potentially multiple highlight objects        
        _, contours, _ = cv2.findContours(hlimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        segm = {}
        for c in contours[:-1]:
            bbox = [None, None, None, None]
            rect = cv2.boundingRect(c)
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            hlbool = np.zeros_like(hlimg).astype(np.uint8)
            cv2.drawContours(hlbool, [c], 0, (255,255,255), cv2.FILLED)
            segm = encode(np.asarray(hlbool, order='F'))
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segm,
                "category_id": 1, # highlight
            }
            record['annotations'].append(obj)

        # potentially multiple wrinkle objects        
        _, contours, _ = cv2.findContours(wrimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        segm = {}
        for c in contours[:-1]:
            bbox = [None, None, None, None]
            rect = cv2.boundingRect(c)
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            wrbool = np.zeros_like(wrimg).astype(np.uint8)
            cv2.drawContours(wrbool, [c], 0, (255,255,255), cv2.FILLED)
            segm = encode(np.asarray(wrbool, order='F'))
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segm,
                "category_id": 2, # wrinkle
            }
            record['annotations'].append(obj)

        dataset_dicts.append(record)
    return dataset_dicts


def tuple_names(impath, gtpath_iris, gtpath_hl, gtpath_wr, istraining=False):
    """
    Generate a quadruplet of file names
    """
    names = []

    # if we're training, use the HL/WR ground truth as guideline
    if istraining:
        files_hl = glob(f"{gtpath_hl}/*")
        files_wr = glob(f"{gtpath_wr}/*")
        allids = list(set([re.findall(r'[0-9]{4}_[LR]_[0-9]+_[0-9]+', a)[0] for a in files_hl+files_wr]))
    else:
        # if it's only for prediction, we don't care whether GT exists or not.
        fim = glob(f'{impath}/*')
        # allids = list(set([re.findall(r'[0-9]{4}_?[LR][_IG]*_[0-9]+_[0-9]+|S[0-9]{4}[LR][0-9]{2}|[0-9]{5}d[0-9]+|C[0-9]+_S[0-9]_I[0-9]+|u[0-9]{4}s[0-9]{4}_ir_[lr]_[0-9]{4}|[0-9]{4}[lr][0-9]{4}', a)[0] for a in fim]))
        allids = []
        for a in fim:
            fn = re.findall(r'[0-9]{4}_?[LR][_IG]*_[0-9]+_[0-9r]+|S[0-9]{4}[LR][0-9]{2}|[0-9]{5}d[0-9]+|C[0-9]+_S[0-9]_I[0-9]+|u[0-9]{4}s[0-9]{4}_ir_[lr]_[0-9]{4}|[0-9]{4}[lr][0-9]{4}', a)
            if len(fn)>0:
                allids.append(fn[0])
            else:
                print(f"Mask not found: {a}")
        allids = list(set(allids))

    for fid in allids:
        fim = glob(f'{impath}/*{fid}*')[0]
        
        fgt, fhl, fwr = [], [], []
        if istraining:
            fgt = glob(f'{gtpath_iris}/{fid}*')
            if fgt:
                fgt = fgt[0]
            fhl = glob(f'{gtpath_hl}/{fid}*')
            if fhl:
                fhl = fhl[0]
            fwr = glob(f'{gtpath_wr}/{fid}*')
            if fwr:
                fwr = fwr[0]
        names.append((fim, fgt, fhl, fwr))
    return names


def init_detectron(weightsfile, lr=0.00025, mi=5000, 
    impath='./samples/warsaw/images_nir',
    gtpath_iris='./samples/warsaw/nir-manual-seg',
    gtpath_hl='./samples/warsaw/highlight-results',
    gtpath_wr='./samples/warsaw/wrinkles-results',
    istraining=False,
    ):
    maskrcnn = MaskRCNN()

    trainfiles = tuple_names(impath, gtpath_iris, gtpath_hl, gtpath_wr, istraining)

    maskrcnn.TRAIN_IMGNAMES = [r[0] for r in trainfiles]
    maskrcnn.TRAIN_MSKNAMES = [r[1] for r in trainfiles]
    maskrcnn.TRAIN_HLNAMES  = [r[2] for r in trainfiles]
    maskrcnn.TRAIN_WRNAMES  = [r[3] for r in trainfiles]

    # LOAD DATASET DICTIONARIES
    maskrcnn.trainds_dicts = get_dicts(maskrcnn.TRAIN_IMGNAMES, 
                                       maskrcnn.TRAIN_MSKNAMES,
                                       maskrcnn.TRAIN_HLNAMES,
                                       maskrcnn.TRAIN_WRNAMES,
                                       istraining)
    print('Loaded TRAIN dataset metadata.')

    # REGISTER DATASETS
    DatasetCatalog.register("hl_train", lambda i=maskrcnn.TRAIN_IMGNAMES,
                                               m=maskrcnn.TRAIN_MSKNAMES,
                                               h=maskrcnn.TRAIN_HLNAMES,
                                               w=maskrcnn.TRAIN_WRNAMES: get_dicts(i, m, h, w))
    MetadataCatalog.get('hl_train').set(stuff_classes=["iris","highlights","wrinkles"])
    maskrcnn.iris_metadata = MetadataCatalog.get("hl_train")

    # SET UP MASK-RCNN
    maskrcnn.cfg = get_cfg()
    maskrcnn.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    maskrcnn.cfg.DATASETS.TRAIN = ("hl_train",)
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
    maskrcnn.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
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

def plot_image(maskrcnn, file_dict,):
    img = cv2.imread(file_dict['file_name'])
    visualizer = Visualizer(img[:,:, ::-1], metadata=maskrcnn.iris_metadata)
    vis = visualizer.draw_dataset_dict(file_dict['file_name'])
    plt.figure()
    plt.imshow(vis.get_image())
    plt.title(str(file_dict["image_id"])+" "+file_dict['file_name'])
    sname = os.path.basename(file_dict['file_name']).split('.')[0]
    plt.savefig(f"tmp_{sname}.png")

def show_training_samples(maskrcnn):
    for d in random.sample(maskrcnn.trainds_dicts, 3):
        plot_image(maskrcnn, d)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, help='Specify the file from which initial weights should be loaded. Default is ResNet(ImageNet).')
    ap.add_argument('--impath',type=str, help='Override path for training/prediction images.')
    ap.add_argument('--gtpath_iris',type=str, help='Override path for training iris ground truth masks.')
    ap.add_argument('--gtpath_hl', type=str, help='Override path for training highlight ground truth masks.')
    ap.add_argument('--gtpath_wr', type=str, help='Override path for training wrinkles ground truth masks.')
    ap.add_argument('--lr', type=float, default=0.00025, help='Learning rate.')
    ap.add_argument('--mi', type=int, default=5000, help='Max iterations.')
    ap.add_argument('--savemodel', type=str, help='Path to file where to save the newly trained model.')
    ap.add_argument('--saveexamples', dest='saveexamples', action='store_true')
    ap.set_defaults(saveexamples=False)
    ap.add_argument('--savepredictions', dest='savepreds', action='store_true')
    ap.set_defaults(savepreds=False)
    ap.add_argument('--output', type=str, help='Output prefix.')
    ap.add_argument('--genmasks', choices=['combined','individual'], help='Options to save predicted masks.')
    args = ap.parse_args()

    initargs = {k:v for k, v in vars(args).items() if (any(x in k for x in ['path'])) and v is not None}
    maskrcnn = init_detectron(
        weightsfile=args.weights if args.weights else None,
        lr=args.lr,
        mi=args.mi, 
        istraining=((not args.savepreds) and (args.genmasks is None)),
        **initargs
    )
    maskrcnn.cfg.MODEL.DEVICE = "cpu"

    if args.saveexamples:
        show_training_samples(maskrcnn)

    if args.savepreds:
        maskrcnn.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
        maskrcnn.cfg.DATASETS.TEST = ("hl_train", )
        predictor = DefaultPredictor(maskrcnn.cfg)
        for d in random.sample(maskrcnn.trainds_dicts, 5):
            im = cv2.imread(d['file_name'])
            outputs = predictor(im)
            v = Visualizer(
                im[:,:,::-1],
                metadata=maskrcnn.iris_metadata,
                scale=1.,
                instance_mode=ColorMode.IMAGE_BW
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize=(12,8))
            plt.imshow(v.get_image())
            sname = os.path.basename(d['file_name']).split('.')[0]
            plt.savefig(f"tmp_{sname}.png")
    if args.genmasks:
        outdir = 'predictions'
        if args.output:
            outdir = args.output
        os.makedirs(outdir,exist_ok=True)

        maskrcnn.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
        maskrcnn.cfg.DATASETS.TEST = ("hl_train", )
        predictor = DefaultPredictor(maskrcnn.cfg)

        for d in maskrcnn.trainds_dicts:
            im = cv2.imread(d['file_name'])
            sname = os.path.basename(d['file_name']).split('.')[0]
            outputs = predictor(im)
            predclasses = outputs["instances"].pred_classes.cpu().numpy()
            predshape = outputs["instances"].pred_masks.cpu().numpy().shape[0]
            predmasks = outputs["instances"].pred_masks.cpu().numpy().astype(np.uint8)

            irismask = np.sum(predmasks[np.where(predclasses==0)[0],...], axis=0)*255
            hlmask = np.sum(predmasks[np.where(predclasses==1)[0],...], axis=0)*255
            wrmask = np.sum(predmasks[np.where(predclasses==2)[0]], axis=0)*255

            # save individual masks
            if args.genmasks=='individual':
                cv2.imwrite(
                    f"{outdir}/{sname}_iris.png",
                    irismask
                )
                cv2.imwrite(
                    f"{outdir}/{sname}_highlights.png",
                    hlmask
                )
                cv2.imwrite(
                    f"{outdir}/{sname}_wrinkles.png",
                    wrmask
                )
            # combine masks into a single one
            elif args.genmasks=='combined':
                cv2.imwrite(
                    f"{outdir}/{sname}_combined.png",
                    irismask - hlmask - wrmask
                )

            print(sname)
    else:
        train(maskrcnn, savemodel='' if not args.savemodel else args.savemodel)
