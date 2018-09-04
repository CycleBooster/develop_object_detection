import keras
import numpy as np
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPool2D,Activation,BatchNormalization
def build_vgg():
    vgg=VGG16(weights='imagenet',include_top=True)

    #start building model
    x=Input(shape=(None,None,3))
    y=Conv2D(64,[3,3],padding='same',name="block1_conv1")(x)
    y=BatchNormalization(name="block1_bn1")(y)
    y=Activation('relu',name="block1_act1")(y)
    y=Conv2D(64,[3,3],padding='same',name="block1_conv2")(y)
    y=BatchNormalization(name="block1_bn2")(y)
    y=Activation('relu',name="block1_act2")(y)
    y=MaxPool2D(pool_size=(2, 2),name="block1_pool")(y)
    y=Conv2D(128,[3,3],padding='same',name="block2_conv1")(y)
    y=BatchNormalization(name="block2_bn1")(y)
    y=Activation('relu',name="block2_act1")(y)
    y=Conv2D(128,[3,3],padding='same',name="block2_conv2")(y)
    y=BatchNormalization(name="block2_bn2")(y)
    y=Activation('relu',name="block2_act2")(y)
    y=MaxPool2D(pool_size=(2, 2),name="block2_pool")(y)
    y=Conv2D(256,[3,3],padding='same',name="block3_conv1")(y)
    y=BatchNormalization(name="block3_bn1")(y)
    y=Activation('relu',name="block3_act1")(y)
    y=Conv2D(256,[3,3],padding='same',name="block3_conv2")(y)
    y=BatchNormalization(name="block3_bn2")(y)
    y=Activation('relu',name="block3_act2")(y)
    y=Conv2D(256,[3,3],padding='same',name="block3_conv3")(y)
    y=BatchNormalization(name="block3_bn3")(y)
    y=Activation('relu',name="block3_act3")(y)
    y=MaxPool2D(pool_size=(2, 2),name="block3_pool")(y)
    y=Conv2D(512,[3,3],padding='same',name="block4_conv1")(y)
    y=BatchNormalization(name="block4_bn1")(y)
    y=Activation('relu',name="block4_act1")(y)
    y=Conv2D(512,[3,3],padding='same',name="block4_conv2")(y)
    y=BatchNormalization(name="block4_bn2")(y)
    y=Activation('relu',name="block4_act2")(y)
    y=Conv2D(512,[3,3],padding='same',name="block4_conv3")(y)
    y=BatchNormalization(name="block4_bn3")(y)
    y=Activation('relu',name="block4_act3")(y)
    y=MaxPool2D(pool_size=(2, 2),name="block4_pool")(y)
    y=Conv2D(512,[3,3],padding='same',name="block5_conv1")(y)
    y=BatchNormalization(name="block5_bn1")(y)
    y=Activation('relu',name="block5_act1")(y)
    y=Conv2D(512,[3,3],padding='same',name="block5_conv2")(y)
    y=BatchNormalization(name="block5_bn2")(y)
    y=Activation('relu',name="block5_act2")(y)
    y=Conv2D(512,[3,3],padding='same',name="block5_conv3")(y)
    y=BatchNormalization(name="block5_bn3")(y)
    y=Activation('relu',name="block5_act3")(y)
    y=MaxPool2D(pool_size=(2, 2),name="block5_pool")(y)
    model=Model(inputs=x,outputs=y)

    #load weight
    name_list=["block1_conv1","block1_conv2",
                "block2_conv1","block2_conv2",
                "block3_conv1","block3_conv2","block3_conv3",
                "block4_conv1","block4_conv2","block4_conv3",
                "block5_conv1","block5_conv2","block5_conv3"]
    for name in name_list:
        weight=vgg.get_layer(name).get_weights()
        model.get_layer(name).set_weights(weight)
    model.save("./model/vgg_pretrain.h5")