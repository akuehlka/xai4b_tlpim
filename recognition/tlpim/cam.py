import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
import cv2
import os
import argparse


class CAMHelper():

    input_shape = (224,224,3)

    def __init__(self, 
            weights='/models/triplet-postmortem.h5py',
            images_dir='/workdir/maskrcnn/cropped',
            output_dir='/workdir/tlpim/cam'
        ):
        # weights for the prediction model
        self.model = self._get_prediction_model(weights)
        self.images_dir = images_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def _get_prediction_model(self, weights):
        # load the complete triplet loss model
        tlm = load_model(weights, compile=False)

        # strip the backbone
        submodel_layer = tlm.get_layer('vggface_resnet_enc')
        model = keras.Model(submodel_layer.inputs, submodel_layer.outputs)
        return model


    def load_sample(self, images_dir, image_name):
        impath = os.path.join(images_dir,image_name)
        if not os.path.exists(impath):
            # try adding a subdir for subject
            impath = os.path.join(images_dir, image_name[0:6],image_name)
        image = cv2.imread(impath)
        assert image is not None, f"Error loading image {impath}"
        if image.ndim==2:
            image = np.stack((image, image), axis=2)

        if image.shape != self.input_shape:
            image = cv2.resize(image, self.input_shape[:2])
        assert image.shape == self.input_shape, "Image shape does not match network input."

        return image



    def gen_heatmap(self, input_sample):
        # calculate the class activation map
        last_conv_layer = self.model.get_layer('conv5_1_1x1_proj')
        grads = K.gradients(self.model.output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        
        iterate = K.function([self.model.input],[pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output = iterate([input_sample])
        
        conv_layer_output[:,:] *= pooled_grads_value
        heatmap = np.mean(conv_layer_output,axis=-1)
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        upsample = cv2.resize(heatmap, (224,224)) 

        return upsample

    
    @staticmethod
    def process(input_images, weights, images_dir, output_dir):
        camh = CAMHelper(
            weights=weights,
            images_dir=images_dir,
            output_dir=output_dir
        )
        filenames = []
        for input_image in input_images:
            ext = os.path.splitext(input_image)[1]
            filename = input_image.replace(ext, '_heatmap.png')
            filename = os.path.join(camh.output_dir, filename)

            if not os.path.exists(filename):
                image = camh.load_sample(camh.images_dir, input_image)
                image = image[np.newaxis, ...]
                heat_img = camh.gen_heatmap(image)
                heat_img = (heat_img*255).astype(np.uint8)

                cv2.imwrite(filename, heat_img)
            
            filenames.append(filename)
        
        return filenames
        
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images',type=str,nargs='+')
    args = ap.parse_args()
    
    output = CAMHelper.process(args.images)    

    print(output)
    