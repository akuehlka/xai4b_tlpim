from keras import Model
from keras_vggface import VGGFace # https://github.com/rcmalli/keras-vggface
from keras.layers import Dense, GlobalAveragePooling2D

def resnet_model(shape, n_subjects):

    vggface_model = VGGFace(
        include_top=False,
        model='resnet50',
        weights='vggface',
        input_shape=shape,
        classes=n_subjects,
    )

    last_layer = vggface_model.get_layer('avg_pool').output
    x = GlobalAveragePooling2D()(last_layer)
    classifier = Dense(n_subjects, activation='softmax', name='fc8')(x)

    resnet_model = Model(vggface_model.input, outputs=[classifier])

    return resnet_model

