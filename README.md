# WACV 2022 Supplementary material - Source Code
# Submission ID #1262

---
## Requirements

- Docker version 20.10.7 or higher

---
## Instructions

### Get a copy
Clone this repository into your machine:
```
git clone https://...
```

### Build the container
In the folder where you cloned the repository, run the following command to build the container with the environment to run TLPIM:
```
docker build --build-arg USER_ID=$UID -t tlpim-cpu . 
```
Building the container for the first time can take a while.

### Download the models
Download the trained models for Mask R-CNN and TLPIM from these addresses:

[TLPIM](https://notredame.box.com/s/pn2woicmkhjqdcnzohvjltlntr8qr79c)

[Mask R-CNN](https://notredame.box.com/s/dnhqy7v32jhpv4qlzfbj2scbzbdxarvo)

Place the downloaded files on the `models` folder of the project.

### Run the container
After building the container, you can run it with:
```
docker run --rm -ti tlpim-cpu
```
The remaining tasks described in this README should be executed inside the `tlpim-cpu` container.

---
## Iris Segmentation

### Activating the environment
To activate the detectron2 environment, run: 
``` 
conda activate detectron2
```
This command is necessary before running any segmentation task.

### Run segmentation
Command to run segmentation on the provided samples: 
```
python segmentation/maskrcnn/iris/finetune.py \
--weights models/maskrcnn-pm.pth \
--impath samples/ \
--genmasks individual \
--output ./
```

### Train Mask R-CNN
To train Mask R-CNN on iris images, use `segmentation/maskrcnn/iris/train_maskrcnn.py`. 

Check possible arguments with `python segmentation/maskrcnn/iris/train_maskrcnn.py --help`.

### Fine tune Mask R-CNN
To train Mask R-CNN on highlights and wrinkles, use `segmentation/maskrcnn/iris/finetune.py`. 

Check possible arguments with `python segmentation/maskrcnn/iris/finetune.py --help`.

---
## Iris Recognition

### Activate the environment
To activate TLPIM environment, run inside the container: 
``` 
conda activate tlpim
```
This command is necessary before running any recognition task.

### Run TLPIM
To use TLPIM to perform iris comparison use a command like this:
```
python  recognition/tlpim/evaluation.py predict \
--outpath ./ \
--weights ./models/triplet-postmortem.h5 \
--image_folder ./samples/cropped/ \
--testfiles ./samples/filelist.txt
```

### Train TLPIM
You can train your own network using a command like this:
```
python recognition/tlpim/triplet_train.py \
-o ./ \
-p test- \
--type resnet \
-tf ./samples \
-mf ./ \
--lr 0.00001 \
--solver adam \
--batch_size 24 \
-vp 20 \
--min_iterations 20000 \
--patience 20 \
--update_base_model
```
Please note that training on the samples provided with this code will result in error, because there are not enough samples for formation of triplets, or even train/test partitioning.

Use `python recognition/tlpim/triplet_train.py --help` to view available arguments.

---
## Acknowledgements 
Parts of this code were based on the code kindly provided by the authors of https://doi.org/10.1109/IJCB48548.2020.9304939