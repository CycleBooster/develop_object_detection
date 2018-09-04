import keras
import tensorflow as tf
import numpy as np
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing import image
from keras.layers import Conv2D,BatchNormalization,Activation,Dense,Input,Add
from keras.layers import Flatten,AveragePooling2D,MaxPooling2D,Reshape,concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.losses import categorical_crossentropy,mse,binary_crossentropy,mean_absolute_error
from keras.callbacks import LearningRateScheduler,TensorBoard
from keras.applications import ResNet50
import keras.backend as K
import cv2
class cifar():
    def __init__(self, model_path,lr=0.01):
        self.model_path=model_path
        self.lr=lr
        
        x=Input(shape=(32,32,3))
        temp_layer=Conv2D(64,[3,3],padding='same')(x)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        med_layer=MaxPooling2D(pool_size=(2, 2), padding='valid')(temp_layer)

        temp_layer=Conv2D(64,[3,3],padding='same')(med_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        temp_layer=Conv2D(64,[3,3],padding='same')(temp_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        med_layer=Add()([med_layer,temp_layer])
        temp_layer=Conv2D(64,[3,3],padding='same')(med_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        temp_layer=Conv2D(64,[3,3],padding='same')(temp_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        med_layer=Add()([med_layer,temp_layer])

        med_layer=Conv2D(128,[3,3],padding='same',strides=2)(med_layer)
        med_layer=Activation('relu')(med_layer)
        
        temp_layer=Conv2D(128,[3,3],padding='same')(med_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        temp_layer=Conv2D(128,[3,3],padding='same')(temp_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        med_layer=Add()([med_layer,temp_layer])
        temp_layer=Conv2D(128,[3,3],padding='same')(med_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        temp_layer=Conv2D(128,[3,3],padding='same')(temp_layer)
        temp_layer=BatchNormalization()(temp_layer)
        temp_layer=Activation('relu')(temp_layer)
        med_layer=Add()([med_layer,temp_layer])
        med_layer=AveragePooling2D()(med_layer)
        med_layer=Flatten()(med_layer)
    
        out_layer=Dense(512)(med_layer)
        out_layer=Activation('relu')(out_layer)
        out_layer=Dense(10)(out_layer)
        y=Activation('softmax')(out_layer)
        
        #pretrain_model=ResNet50(weights='imagenet')
        #x=pretrain_model.input
        #pretrain_output=pretrain_model.get_layer("flatten_1").output
        #y=Dense(10,activation='softmax')(pretrain_output)
        self.model=Model(inputs=x,outputs=y)
        Optimizer=SGD(lr=self.lr)
        self.model.compile(optimizer=Optimizer,loss=categorical_crossentropy,metrics=['accuracy'])
        #self.model.summary()

    def train(self,batch_size=32,epoch=10):
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical( y_train, 10)
        y_test=to_categorical( y_test, 10)
        
        #in_train=np.zeros((x_train.shape[0],224,224,3))
        #in_test=np.zeros((x_test.shape[0],224,224,3))
        #for x in x_train:
        #    in_train=cv2.resize(x,(224,224))
        #for x in x_test:
        #    in_testcv2.resize(x,(224,224))
        #print(in_train.shape)
        #x_train=preprocess_input(x_train)
        #x_test=preprocess_input(x_test)
        lr_changer=LearningRateScheduler(self.lr_scheduler)
        self.model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=epoch,batch_size=batch_size,callbacks=[lr_changer])


    def lr_scheduler(self,epoch):
        start_lr=self.lr
        power=(int)(epoch/3)
        print("lr=",start_lr*(0.5**power))
        return start_lr*(0.5**power)
   

class image_callback(keras.callbacks.Callback):
    def __init__(self,pred):
        super(image_callback,self).__init__()
        self.pred=pred
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    
    x=x.astype('float')
    if dim_ordering == 'th':
        x[:,0, :, :] -= 103.939
        x[:,1, :, :] -= 116.779
        x[:,2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:,::-1, :, :]
        assert x.shape[1]==3
    else:
        x[:,:, :, 0] -= 103.939
        x[:,:, :, 1] -= 116.779
        x[:,:, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:,:, :, ::-1]
        assert x.shape[3]==3
    return x
