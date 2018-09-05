import keras
import sys
import tensorflow as tf
import numpy as np
import math
from keras.applications import ResNet50,VGG16
from keras.preprocessing import image
from keras.layers import Conv2D,Activation,Lambda,LeakyReLU
from keras.layers import Concatenate,Input,Add,Conv2DTranspose,Reshape,UpSampling2D
from keras.optimizers import SGD,Adam
from keras.models import Model,load_model
from keras.losses import categorical_crossentropy,binary_crossentropy,mean_absolute_error,mse
from keras.initializers import Constant,RandomUniform,RandomNormal
from keras.constraints import non_neg
from keras.callbacks import LearningRateScheduler,TensorBoard,LambdaCallback
from keras import regularizers
import keras.backend as K
from math import log
from data import *
from setup import *
from data_generator import data_generator
from dataset import data_reader
import cv2
from progress import bar
cluster_thre=0.5
weight_decay_rate=0.0001
class ObjectDetector():
    def __init__(self, model_path,center_flag,lr=0.001,pred_list=[],test_size=(384,384)):
        self.model_path=model_path
        self.lr=lr
        self.test_size=test_size
        self.center_flag=center_flag
        box_list,grid_depth_list=self.initial_box_list()
        self.grid_depth_list=grid_depth_list
        self.data_grid_depth_list=grid_depth_list
        self.origin_id=[i for i,grid_gepth in enumerate(grid_depth_list) if grid_gepth>0]
        self.true_depth=len(box_list)
        self.true_box=np.reshape(box_list,[1,1,1,self.true_depth,2])
        self.pred_list=pred_list
        self.op_name=[]
        
        pretrain_model=ResNet50(input_shape=(None,None,3),weights='imagenet',include_top=False)
        pretrain_output=pretrain_model.get_layer("activation_49").output
        x=pretrain_model.input
        feature_model=Model(inputs=x,outputs=pretrain_output)
        layer_name=["activation_49","activation_40","activation_22"]
        output_y_list=self.output_layer(feature_model,self.grid_depth_list,layer_name)
        for output_y in output_y_list:
            self.op_name.append(output_y.op.name)
        self.opnametoOriginID={name:self.origin_id[index] for index,name in enumerate(self.op_name)}
        self.opnametoID={name:index for index,name in enumerate(self.op_name)}
        self.optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model=Model(inputs=x,outputs=output_y_list)

        self.grid_depth_list=[i for i in self.grid_depth_list if i > 0]
        
        self.model.compile(optimizer=self.optimizer,loss=self.loss_fun,metrics=[self.conf_acc,self.prec,self.recall_1,self.recall_3,self.recall_5,self.box_acc])

        # test_name_list=['temp_conf_0','temp_conf_1']
        # for test_name in test_name_list:
        #     test_layer=self.model.get_layer(test_name)
        #     test_weight=test_layer.get_weights()
        #     flat_weight=test_weight[0].flatten()
        #     print(np.amin(flat_weight))
        #     print(np.amax(flat_weight))
    def initial_box_list(self):
        base_box_list=[[1,1],[1,2],[2,1]]
        ratio_in_layer=[1,1.26,1.59]
        # layer_base=[128,64,32]
        layer_base=[128,64]
        out_box_list=[]
        out_grid_depth_list=[]
        for L_base in layer_base:
            temp_box_list=[]
            for ratio in ratio_in_layer:
                temp_box_list=temp_box_list+[[box[0]*L_base*ratio,box[1]*L_base*ratio] for box in base_box_list]
            out_grid_depth_list.append(len(temp_box_list))
            out_box_list=out_box_list+temp_box_list
        out_box_list=np.array(out_box_list,dtype=float)
        return out_box_list,out_grid_depth_list
    def output_layer(self,feature_model,grid_depth_list,layer_name):
        top_depth=1024
        output_list=[]
        last_feature_layer=None
        for i in range(len(grid_depth_list)):
            #rebuild feature
            feature_size=256#(int)(top_depth/(2**i))
            now_feature_layer=feature_model.get_layer(layer_name[i]).output
            now_feature_layer=Conv2D(feature_size,[1,1],strides=1,padding='same',name="in_conv_"+str(i)
                ,kernel_regularizer=regularizers.l2(weight_decay_rate))(now_feature_layer)
            if last_feature_layer!=None:
                last_feature_layer=UpSampling2D(size=[2,2],name="up_sample_"+str(i))(last_feature_layer)
                # last_feature_layer=Conv2DTranspose(feature_size,[3,3],strides=2,padding='same',name="tp_conv_"+str(i)
                #     ,kernel_regularizer=regularizers.l2(weight_decay_rate))(last_feature_layer)
                now_feature_layer=Add(name="tp_add_"+str(i))([now_feature_layer,last_feature_layer])
            # build output
            if grid_depth_list[i]>0:
                last_feature_layer=now_feature_layer
                out_feature=Conv2D(feature_size,[3,3],strides=1,padding='same',name="out_feature_"+str(i)
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(now_feature_layer)
                #category subnet
                temp_cat_y=self.normal_layer(out_feature,feature_size,loop=4)
                cat_y=Conv2D(class_width,[3,3],padding='same',name="cat_"+str(i)
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_cat_y)
                cat_y=Lambda(self.cat_act,arguments={"grid_depth":grid_depth_list[i]})(cat_y)

                shared_conf_y=Conv2D(1,[3,3],padding='same',name="shared_conf_"+str(i)
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_cat_y)
                #location subnet
                temp_loc_y=self.normal_layer(out_feature,feature_size,loop=4)
                box_y=Conv2D(4*grid_depth_list[i],[3,3],padding='same',name="box_"+str(i)
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_loc_y)
                box_y=Lambda(self.reshape_layer,arguments={"grid_depth":grid_depth_list[i]})(box_y)
                box_y=Lambda(self.box_act)(box_y)

                neg_conf_y=Conv2D(grid_depth_list[i],[3,3],padding='same',name="neg_conf_"+str(i)
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_loc_y)
                neg_conf_y=Lambda(self.reshape_layer,arguments={"grid_depth":grid_depth_list[i]})(neg_conf_y)
                #output operation
                conf_y=Lambda(self.conf_act,arguments={"shared_conf":shared_conf_y})(neg_conf_y)

                output_y=Concatenate(name="output_y_"+str(i))([conf_y,cat_y,box_y])
                output_list.append(output_y)

        return output_list
    def conf_act(self,x,shared_conf):
        shared_conf=tf.tanh(shared_conf)
        shared_conf=tf.nn.relu(shared_conf)
        neg_conf=tf.tanh(x)
        neg_conf=tf.nn.relu(neg_conf)
        base_tensor=tf.ones(tf.shape(neg_conf))
        temp_shared_conf=tf.expand_dims(shared_conf,axis=-2)
        temp_shared_conf=temp_shared_conf*base_tensor
        out_conf=temp_shared_conf-neg_conf
        return out_conf
    def cat_act(self,x,grid_depth):
        x=tf.nn.softmax(x)
        y_shape=tf.shape(x)
        base_shape=tf.concat([y_shape[:3],[grid_depth,y_shape[3]]],axis=-1)
        base_tensor=tf.ones(base_shape)
        out_cat=tf.expand_dims(x,axis=-2)
        out_cat=out_cat*base_tensor
        return out_cat
    def box_act(self,x):
        if self.center_flag:
            box_xy,box_wh=tf.split(x,[2,2],axis=-1)
            box_xy=tf.sigmoid(box_xy)
            box_wh=tf.tanh(box_wh)
            box_y=tf.concat([box_xy,box_wh],axis=-1)
        else:
            box_y=tf.tanh(x)
        return box_y
    
    def normal_layer(self,x,layer_depth,loop=1):#to train confidence, initialize bias to 0
        temp_x=x
        for i in range(loop):
            temp_x=Conv2D(layer_depth,[3,3],padding='same',kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_x)
            temp_x=Activation("relu")(temp_x)
        return temp_x
    def reshape_layer(self,x,grid_depth):
        y_shape=tf.shape(x)
        last_depth=tf.to_int32(y_shape[-1]/grid_depth)
        out_shape=tf.concat([y_shape[:3],[grid_depth,last_depth]],axis=-1)
        y=tf.reshape(x,shape=out_shape)
        return y

    def get_mask(self,y_true,y_pred,debug_id=-1,get_answer=False):
        if debug_id!=-1:
            id=debug_id
        else:
            id=self.opnametoID[y_pred.op.name]
        box_true=y_true[1:5]
        box_pred=y_pred[1:5]
        IOU=self.tf_getIOU(box_true,box_pred,y_pred,cluster=True,debug_id=debug_id)
        IOU_maxarg=tf.argmax(IOU,axis=-1)
        IOU_mask=tf.one_hot(IOU_maxarg,depth=self.grid_depth_list[id])
        IOU_mask=tf.expand_dims(IOU_mask,axis=-1)
        max_IOU=tf.reduce_max(IOU,axis=-1)
        temp_true=tf.expand_dims(y_true,axis=-3)
        temp_true=temp_true*IOU_mask
        out_true=tf.reduce_max(temp_true,axis=-2)
        if get_answer:
            out_true=out_true*tf.to_float(tf.expand_dims(max_IOU,axis=-1)>0.5)
        out_true=tf.stop_gradient(out_true)
        max_IOU=tf.stop_gradient(max_IOU)
        return out_true,max_IOU

    def smooth_l1_loss(self,label,pred):#0<=label<=1
        loss=tf.abs(label-pred)-0.25
        l2_loss=(label-pred)*(label-pred)
        loss=tf.where(tf.abs(label-pred)>0.5,loss,l2_loss)
        return tf.reduce_mean(loss,axis=-1)
    def focal_loss(self,label,pred,focal_factor,pos_weight):
        # focal_loss=0
        # focal_loss=focal_loss-label*(1-pred)**focal_factor*tf.log(tf.clip_by_value(pred,1e-10,1.0))*pos_weight
        # focal_loss=focal_loss-(1-label)*(pred)**focal_factor*tf.log(tf.clip_by_value(1-pred,1e-10,1.0))*(1-pos_weight)
        pos_loss=tf.where(label>=pred,-tf.clip_by_value(label-pred,1e-10,1.0)**focal_factor*tf.log(tf.clip_by_value(pred+1-label,1e-10,1.0))*pos_weight,tf.zeros(tf.shape(label)))
        pos_loss=tf.where(pred<1,pos_loss,tf.zeros(tf.shape(pos_loss)))
        neg_loss=tf.where(label<pred,-tf.clip_by_value(pred-label,1e-10,1.0)**focal_factor*tf.log(tf.clip_by_value(1-pred+label,1e-10,1.0))*(1-pos_weight),tf.zeros(tf.shape(label)))
        neg_loss=tf.where(pred>0,neg_loss,tf.zeros(tf.shape(neg_loss)))
        focal_loss=pos_loss+neg_loss
        return focal_loss
    def loss_fun(self,y_true,y_pred,small_print=False,debug_id=-1,debug_origin_id=0):
        if small_print==True:
            # y_true=tf.constant(y_true,dtype=tf.float32)
            # y_pred=tf.constant(y_pred,dtype=tf.float32)
            id=debug_id
        else:
            id=self.opnametoID[y_pred.op.name]
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,debug_id=debug_id,get_answer=False)
        
        conf_true,box_true=tf.split(y_true,[class_width,4],axis=-1)
        conf_pred,box_pred=tf.split(y_pred,[class_width,4],axis=-1)
        total_conf_true=conf_true
        conf_true=conf_true*tf.expand_dims(tf.to_float(IOU_cluster>0.5),axis=-1)
        #get mask
        pos_mask=tf.expand_dims(tf.to_float(IOU_cluster>0.5),axis=-1)*tf.ones(tf.shape(total_conf_true))
        # pos_conf_mask=tf.to_float(total_conf_true>=1)*pos_mask
        # ignore_mask=tf.to_float(total_conf_true>=1)*tf.expand_dims(tf.to_float(IOU_cluster>0.4),axis=-1)
        pos_conf_mask=tf.to_float(total_conf_true>0)*pos_mask
        ignore_mask=tf.to_float(total_conf_true>0)*tf.expand_dims(tf.to_float(IOU_cluster>0.4),axis=-1)
        neg_conf_mask=tf.where(ignore_mask>0,tf.zeros(tf.shape(pos_conf_mask)),(1-pos_conf_mask))
        # neg_conf_mask=tf.where(pos_mask>0,neg_conf_mask*(1-total_conf_true),neg_conf_mask)
        conf_mask=pos_conf_mask+neg_conf_mask
        box_mask=tf.to_float(tf.reduce_max(conf_true,axis=-1)>0)
        # box_mask=pos_conf_mask
        if small_print==True:
            # temp_conf_true=tf.expand_dims(mask,axis=-1)
            conf_true=tf.to_float(conf_true>=1)
            temp_y_true=tf.concat([conf_true,box_true],axis=-1)
            return temp_y_true,IOU_cluster
        #adjust box for L1 or L2 loss
        box_base=tf.to_float(self.true_box)
        box_base=tf.split(box_base,self.grid_depth_list,axis=-2)[id]
        w_base,h_base=tf.split(box_base,[1,1],axis=-1)

        if is_center:
            centerx_true,centery_true,width_true,height_true=tf.split(box_true,[1,1,1,1],axis=-1)
            temp_w_true=tf.clip_by_value(width_true/w_base,1e-10,width_true/w_base)
            temp_h_true=tf.clip_by_value(height_true/h_base,1e-10,height_true/h_base)
            width_true,height_true=tf.log(temp_w_true),tf.log(temp_h_true)
            box_true=tf.concat([centerx_true,centery_true,width_true,height_true],axis=-1)
        else:
            true_L,true_R,true_T,true_B=tf.split(box_true,4,axis=-1)
            temp_L_true=tf.clip_by_value(true_L/w_base,1e-10,true_L/w_base)
            temp_R_true=tf.clip_by_value(true_R/w_base,1e-10,true_R/w_base)
            temp_T_true=tf.clip_by_value(true_T/h_base,1e-10,true_T/h_base)
            temp_B_true=tf.clip_by_value(true_B/h_base,1e-10,true_B/h_base)
            true_L,true_R=tf.log(temp_L_true*2),tf.log(temp_R_true*2)
            true_T,true_B=tf.log(temp_T_true*2),tf.log(temp_B_true*2)
            box_true=tf.concat([true_L,true_R,true_T,true_B],axis=-1)

        #get count
        conf_count = tf.reduce_sum(tf.to_float(conf_true>=1))#used pos num sum
        conf_count=tf.where(conf_count>1,conf_count,1)
        box_count = tf.reduce_sum(box_mask)
        #get loss
        # conf_true=tf.to_float(conf_true>=1)
        conf_loss=tf.reduce_sum(self.focal_loss(conf_true,conf_pred,2,0.25)*conf_mask)/(conf_count+1e-6)
        box_loss=tf.reduce_sum(self.smooth_l1_loss(box_true,box_pred)*box_mask)/(box_count+1e-6)
        # box_loss=tf.reduce_sum((1-self.tf_getIOU(box_true,box_pred,y_pred))**2*box_mask)/(box_count+1e-6)
        total_loss=conf_loss+box_loss
        return total_loss
    def debug_IOU_mask(self,test_length=1):
        data_gen=data_generator(self.data_grid_depth_list,"train_total",final_size=7,batch_size=1)
        # data_gen=data_generator(self.data_grid_depth_list,"validate",final_size=7,end_name="_dir",batch_size=1)
        generator=data_gen.generator()
        # input_set,answer_list=data_gen.static_data()
        # in_input=input_set[index:index+test_length]
        # in_answer=answer_list[0][index:index+test_length]
        for i in range(test_length):
            input,answer_list=next(generator)
            in_answer=answer_list[0]
            answer,IOU=self.loss_fun(in_answer,in_answer,small_print=True,debug_id=0)
            debug_fun=K.function([K.learning_phase()],[answer,IOU])
            answer=debug_fun([0])[0]
            IOU=debug_fun([0])[1]
            answer=[answer]
            # print(IOU[0])
            show_answer_list(answer,input,file_save=False)

    def prec(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=y_true[...,:class_width]
        conf_true=tf.to_float(conf_true>=1)
        conf_pred=y_pred[...,:class_width]
        conf_pred=tf.to_float(conf_pred>pos_thre)
        pos_sum=tf.reduce_sum(conf_pred)
        pos_true_sum=tf.reduce_sum(conf_pred*conf_true)
        acc=tf.where(pos_sum>0,100*pos_true_sum/pos_sum,100*tf.ones(tf.shape(pos_sum)))
        return acc
    def recall_1(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=y_true[...,:class_width]
        conf_true=tf.to_float(conf_true>=1)
        conf_pred=y_pred[...,:class_width]
        conf_pred=tf.to_float(conf_pred>0.1)
        true_sum=tf.reduce_sum(conf_true)
        pos_true_sum=tf.reduce_sum(conf_pred*conf_true)
        acc=tf.where(true_sum>0,100*pos_true_sum/true_sum,100*tf.ones(tf.shape(true_sum)))
        return acc
    def recall_3(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=y_true[...,:class_width]
        conf_true=tf.to_float(conf_true>=1)
        conf_pred=y_pred[...,:class_width]
        conf_pred=tf.to_float(conf_pred>0.3)
        true_sum=tf.reduce_sum(conf_true)
        pos_true_sum=tf.reduce_sum(conf_pred*conf_true)
        acc=tf.where(true_sum>0,100*pos_true_sum/true_sum,100*tf.ones(tf.shape(true_sum)))
        return acc
    def recall_5(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=y_true[...,:class_width]
        conf_true=tf.to_float(conf_true>=1)
        conf_pred=y_pred[...,:class_width]
        conf_pred=tf.to_float(conf_pred>0.5)
        true_sum=tf.reduce_sum(conf_true)
        pos_true_sum=tf.reduce_sum(conf_pred*conf_true)
        acc=tf.where(true_sum>0,100*pos_true_sum/true_sum,100*tf.ones(tf.shape(true_sum)))
        return acc
    def conf_acc(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=y_true[...,:class_width]
        conf_true=tf.to_float(conf_true>=1)
        conf_pred=y_pred[...,:class_width]
        conf_pred=tf.where(conf_pred>pos_thre,tf.ones(tf.shape(conf_pred)),tf.zeros(tf.shape(conf_pred)))
        each_conf=1-mean_absolute_error(conf_true,conf_pred)
        acc=tf.reduce_mean(100*each_conf)
        return acc
    def clas_acc(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true=tf.to_float(y_true[...,0]>0)
        true_sum=tf.reduce_sum(conf_true)
        clas_true=tf.argmax(y_true[...,1:class_width],axis=-1)
        clas_pred=tf.argmax(y_pred[...,1:class_width],axis=-1)
        cond=tf.equal(clas_true,clas_pred)
        clas_equal=tf.where(cond,tf.ones(tf.shape(clas_pred)),tf.zeros(tf.shape(clas_pred)))
        pos_clas=conf_true*clas_equal
        pred_sum=tf.reduce_sum(pos_clas)
        acc=tf.where(true_sum>0,100*pred_sum/true_sum,100*tf.ones(tf.shape(pred_sum)))
        return acc
    def box_acc(self,y_true, y_pred):
        y_true,IOU_cluster=self.get_mask(y_true,y_pred,get_answer=True)
        conf_true,box_true=tf.split(y_true,[class_width,4],axis=-1)
        conf_pred,box_pred=tf.split(y_pred,[class_width,4],axis=-1)
        conf_true=tf.to_float(tf.reduce_max(conf_true,axis=-1)>0)
        true_sum=tf.reduce_sum(conf_true)
        IOU=self.tf_getIOU(box_true,box_pred,y_pred)
        IOU=tf.reduce_sum(conf_true*IOU)
        acc=tf.where(true_sum>0,100*IOU/true_sum,100*tf.ones(tf.shape(IOU)))
        return acc
    def tf_getIOU(self,box_true, box_pred,y_pred,cluster=False,debug_id=-1):
        if debug_id!=-1:
            id=debug_id
        else:
            id=self.opnametoID[y_pred.op.name]
        grid_width=32/(2**id)
        box_base=tf.to_float(self.true_box)
        box_base=tf.split(box_base,self.grid_depth_list,axis=-2)[id]
        if cluster:
            box_true=tf.expand_dims(box_true,axis=-3)
            box_base=tf.expand_dims(box_base,axis=-2)
            w_base,h_base=tf.split(box_base,[1,1],axis=-1)
            if self.center_flag:
                centerx_true,centery_true,width_true,height_true=tf.split(box_true,[1,1,1,1],axis=-1)
                centerx_pred,centery_pred=centerx_true,centery_true
                width_pred,height_pred=w_base,h_base
            else:
                left_true,right_true,top_true,bottom_true=tf.split(box_true,[1,1,1,1],axis=-1)
                left_pred,right_pred=left_true,w_base-left_true
                top_pred,bottom_pred=top_true,h_base-top_true
        else:
            w_base,h_base=tf.split(box_base,[1,1],axis=-1)
            if self.center_flag:
                centerx_true,centery_true,width_true,height_true=tf.split(box_true,[1,1,1,1],axis=-1)
                centerx_pred,centery_pred,width_pred,height_pred=tf.split(box_pred,[1,1,1,1],axis=-1)
                centerx_pred,centery_pred=grid_width*centerx_pred,grid_width*centery_pred
                centerx_true,centery_true=grid_width*centerx_true,grid_width*centery_true
                width_pred,height_pred=w_base*tf.exp(width_pred),h_base*tf.exp(height_pred)
            else:
                left_true,right_true,top_true,bottom_true=tf.split(box_true,[1,1,1,1],axis=-1)
                left_pred,right_pred,top_pred,bottom_pred=tf.split(box_pred,[1,1,1,1],axis=-1)
                left_pred,right_pred=0.5*w_base*tf.exp(left_pred),0.5*w_base*tf.exp(right_pred)
                top_pred,bottom_pred=0.5*h_base*tf.exp(top_pred),0.5*h_base*tf.exp(bottom_pred)
                # left_pred,right_pred=w_base*left_pred*2,w_base*right_pred*2
                # top_pred,bottom_pred=h_base*top_pred*2,h_base*bottom_pred*2


            
        if self.center_flag:
            inter_x_min=tf.maximum(centerx_true-width_true/2,centerx_pred-width_pred/2)
            inter_x_max=tf.minimum(centerx_true+width_true/2,centerx_pred+width_pred/2)
            inter_y_min=tf.maximum(centery_true-height_true/2,centery_pred-height_pred/2)
            inter_y_max=tf.minimum(centery_true+height_true/2,centery_pred+height_pred/2)
            inter_width=inter_x_max-inter_x_min
            inter_height=inter_y_max-inter_y_min
        else:
            inter_x_min=tf.minimum(left_true,left_pred)
            inter_x_max=tf.minimum(right_true,right_pred)
            inter_y_min=tf.minimum(top_true,top_pred)
            inter_y_max=tf.minimum(bottom_true,bottom_pred)
            inter_width=inter_x_max+inter_x_min
            inter_height=inter_y_max+inter_y_min

        width_zero=tf.less(inter_width,0)
        height_zero=tf.less(inter_height,0)
        both_zero=tf.logical_and(width_zero,height_zero)
        inter=tf.where(both_zero,-inter_width*inter_height,inter_width*inter_height)

        if self.center_flag:
            union=(width_true)*(height_true)+(width_pred)*(height_pred)-inter
        else:
            union=(left_true+right_true)*(top_true+bottom_true)+(left_pred+right_pred)*(top_pred+bottom_pred)-inter

        IOU=tf.squeeze(inter/(union+1e-6),[-1])
        return IOU

    def pred(self,img_name,file_type='jpg',file_save=True):
        self.model.load_weights(self.model_path)
        img_path = './/test_photo//'+img_name+'.'+file_type
        origin_x = cv2.imread(img_path,-1)
        x=cv2.resize(origin_x,self.test_size)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        preds=self.decode_wh(preds)
        show_answer_list(preds,x,name=img_name,file_save=file_save)
    def pred_dataset(self,name):
        self.model.load_weights(self.model_path)
        import pickle
        data_gen=data_generator(self.data_grid_depth_list,"validate",batch_size=16)
        input_set,answer_list=data_gen.static_data()
        preds = self.model.predict(input_set)
        preds=self.decode_wh(preds)
        with open('./model/'+name+'.txt', 'wb') as handle:
            pickle.dump(preds, handle)
        # return preds
    def test_load_pickle(self,name):
        # self.model.load_weights(self.model_path)
        import pickle
        data_gen=data_generator(self.data_grid_depth_list,"validate",batch_size=16)
        input_set,answer_list=data_gen.static_data()
        with open('./model/'+name+'.txt', 'rb') as handle:
            preds = pickle.load(handle)
            show_answer_list(preds,input_set)

    def pred_video(self,video_name,max_size,file_type='mp4',file_save=False):
        self.model.load_weights(self.model_path)
        video_path="./data/video/"+video_name+"."+file_type
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened()== False:
            print("error in opening video")
            return
        frame_count=0
        # save_path='./test_photo/test_video.mp4'
        # out = cv2.VideoWriter(save_path,fourcc, 20.0, (640,480))
        while 1:
            ret, frame = cap.read()
            height,width=frame.shape[0],frame.shape[1]
            if height>=width:
                width=(int)(max_size*width/height/32+0.5)*32
                height=max_size
            else:
                height=(int)(max_size*height/width/32+0.5)*32
                width=max_size
            if ret==False:
                break
            x=cv2.resize(frame,(width,height))
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            preds = self.model.predict(x)
            preds=self.decode_wh(preds)
            show_answer_list(preds,x,name=video_name+'_'+format(frame_count, "04"),wait=False,file_save=file_save,delete_by_IOU=0.5,out_size=(width,height))
            # show_answer_list(preds,x,name=video_name+'_'+str(frame_count),wait=False,file_save=file_save,flatten=False,out_size=(width,height))
            frame_count=frame_count+1
    def train(self,data_name,batch_size=1,epoch=20,keep_train=False):
        if keep_train ==True:
            self.model.load_weights(self.model_path)

        data_gen=data_generator(self.data_grid_depth_list,data_name,batch_size=batch_size
            ,random_size=True,random_mirror=True,random_width=True,shuffle=True)
        input_generator=data_gen.generator()
        step=data_gen.get_max_batch_index()

        lr_changer=LearningRateScheduler(self.lr_scheduler)
        pred_print=LambdaCallback(on_epoch_end=self.pred_print)
        validate=LambdaCallback(on_epoch_end=self.validate)
        TB=TensorBoard()
        print("start train")
        self.model.fit_generator(input_generator,steps_per_epoch=step,epochs=epoch,
            callbacks=[lr_changer,validate,pred_print,TB])
        self.model.save_weights(self.model_path)

        # test_name_list=['temp_conf_0','temp_conf_1']
        # for test_name in test_name_list:
        #     test_layer=self.model.get_layer(test_name)
        #     test_weight=test_layer.get_weights()
        #     flat_weight=test_weight[0].flatten()
        #     print(np.amin(flat_weight))
        #     print(np.amax(flat_weight))
        #     zerocount=0
        #     for weight in flat_weight:
        #         if weight==0:
        #             zerocount=zerocount+1
        #     print(zerocount)
    def pred_print(self,epoch,logs,load_weight=False):
        if load_weight:
            self.model.load_weights(self.model_path)
        for img_name in self.pred_list:
            img_path = './test_photo/'+img_name+'.jpg'
            origin_x = cv2.imread(img_path,-1)
            x=cv2.resize(origin_x,self.test_size)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            preds = self.model.predict(x)
            preds=self.decode_wh(preds)
            if epoch>=0:
                show_answer_list(preds,x,name=img_name+str(epoch),file_save=True,show=False,flatten=False)
                # show_answer_list(preds,x,name=img_name+str(epoch),file_save=True,show=False,delete_by_IOU=0.5)
            else:
                show_answer_list(preds,x,name=img_name,file_save=True,show=False,delete_by_IOU=0.5)
    def validate(self,epoch,logs):
        data_gen=data_generator(self.data_grid_depth_list,"validate",batch_size=16)
        input_set,answer_list=data_gen.static_data()
        eval = self.model.evaluate(x=input_set,y=answer_list,batch_size=8,verbose=0)
        for index,name in enumerate(self.model.metrics_names):
            print("%s:%.4f"%(name,eval[index]),end=" ")
        print()
    def lr_scheduler(self,epoch):
        if epoch==0:
            self.origin_lr=self.lr
        elif epoch==50:
            self.lr=self.origin_lr*0.1
        elif epoch<80:
            self.lr=self.lr*0.97
        now_lr=self.lr
        self.model.save_weights(self.model_path)
        return now_lr
    def test(self,data_name,index=0,test_length=1,file_save=False):
        self.model.load_weights(self.model_path)
        # data_gen=data_generator(self.data_grid_depth_list,data_name,final_size=7,random_size=False,batch_size=1,random_mirror=False,random_width=False)
        # input_set,answer_list=data_gen.static_data()
        # in_input=input_set[index:index+test_length]
        # in_answer=[]
        # for answer in answer_list:
        #     in_answer.append(answer[index:index+test_length])
        # preds=self.model.predict(in_input,batch_size=1)
        # preds=self.decode_wh(preds)
        # # show_IOU_list(preds,in_answer,in_input,file_save=file_save)
        # show_answer_list(preds,in_input,file_save=file_save,show=True,flatten=False)
        # return in_input,in_answer,preds#output the last one
        data_gen=data_generator(self.data_grid_depth_list,data_name,final_size=10,batch_size=1)
        gen=data_gen.test_data(start_index=index)
        for i in range(test_length):
            in_input,in_answer=next(gen)
            preds=self.model.predict(in_input,batch_size=1)
            preds=self.decode_wh(preds)
            # show_IOU_list(preds,in_answer,in_input,file_save=file_save)
            show_answer_list(preds,in_input,show=True,flatten=False)
        return in_input,in_answer,preds#output the last one

    def test_AP(self,data_name,index=0,test_length=1):
        self.model.load_weights(self.model_path)
        data_gen=data_generator(self.data_grid_depth_list,data_name,final_size=7,batch_size=1)
        input_set,answer_list=data_gen.static_data()
        test_size=input_set.shape[1:3]
        in_input=input_set[index:index+test_length]
        in_answer=[]
        for answer in answer_list:
            in_answer.append(answer[index:index+test_length])
        preds=self.model.predict(in_input,batch_size=1)
        preds=self.decode_wh(preds)
        statistics_AP(preds,in_answer,test_size)

    def evaluate(self,data_name,index=0,test_length=1,test_total=False):
        self.model.load_weights(self.model_path)
        data_gen=data_generator(self.data_grid_depth_list,data_name,final_size=7,batch_size=8)
        input_set,answer_list=data_gen.static_data()
        if test_total==False:
            in_input=input_set[index:index+test_length]
            in_answer=[]
            for answer in answer_list:
                in_answer.append(answer[index:index+test_length])
        else:
            in_input,in_answer=input_set,answer_list

        # import time
        # start_time=time.time()
        eval = self.model.evaluate(x=in_input,y=in_answer,batch_size=8)
        # end_time=time.time()
        for index,name in enumerate(self.model.metrics_names):
            print(name,':',eval[index])
        # print("time=",end_time-start_time)
    def virtual_loss(self,x,take_index=0):
        answer,iou=self.loss_fun(x,x,small_print=True,debug_id=take_index)
        return answer
    def build_answer_model(self,answer_shape,take_index):
        input_holder=Input(shape=answer_shape)
        answer=Lambda(self.virtual_loss,arguments={'take_index':take_index})(input_holder)
        out_model=Model(inputs=input_holder,outputs=answer)
        return out_model
    def evaluate_conf(self,data_name,index=0,test_length=1,test_total=False):
        self.model.load_weights(self.model_path)
        data_gen=data_generator(self.data_grid_depth_list,data_name,end_name="_test",final_size=10,batch_size=8)
        input_set,answer_list=data_gen.static_data()
        if test_total==False:
            in_input=input_set[index:index+test_length]
            in_answer=[]
            for answer in answer_list:
                in_answer.append(answer[index:index+test_length])
            answer_list=in_answer
            input_set=in_input
        step=input_set.shape[0]
        print(step)
        print(input_set.shape)
        import time
        time_list=[]
        time_list.append(time.time())#debug
        preds=self.model.predict(input_set,batch_size=8,verbose=1)
        preds=self.decode_wh(preds)
        temp_answer_list=[]
        time_list.append(time.time())#debug
        for layer_index,answer in enumerate(answer_list):
            print(answer.shape)
            answer_model=self.build_answer_model(answer.shape[1:],layer_index)
            answer=answer_model.predict(answer,batch_size=8,verbose=1)
            temp_answer_list.append(answer)
        answer_list=temp_answer_list
        time_list.append(time.time())#debug
        conf_table,number_sum=statistic_conf(answer_list,preds)
        print(conf_table)
        print(number_sum)
        # show_answer_list(preds,in_input,show=True,flatten=False)
        # show_answer_list(answer_list,in_input,show=True,flatten=False)
        time_list.append(time.time())#debug
        for i in range(len(time_list)-1):
            time_cost=time_list[i+1]-time_list[i]
            print(time_cost,end=" ")
        print() 
    def decode_wh(self,preds):
        np_split_list=[sum(self.grid_depth_list[:i+1])for i in range(len(self.grid_depth_list))]
        wh_base_list=np.split(self.true_box,np_split_list,axis=-2)
        out_preds=[]
        if len(self.grid_depth_list)==1:
            preds=[preds]
        for index,scale_pred in enumerate(preds):
            grid_width=32/(2**index)
            w_base,h_base=wh_base_list[index][...,0],wh_base_list[index][...,1]
            if self.center_flag:
                scale_pred[...,0+class_width]=scale_pred[...,0+class_width]
                scale_pred[...,1+class_width]=scale_pred[...,1+class_width]
                scale_pred[...,2+class_width]=w_base*np.exp(scale_pred[...,2+class_width])
                scale_pred[...,3+class_width]=h_base*np.exp(scale_pred[...,3+class_width])
            else:
                scale_pred[...,0+class_width]=0.5*w_base*np.exp(scale_pred[...,0+class_width])
                scale_pred[...,1+class_width]=0.5*w_base*np.exp(scale_pred[...,1+class_width])
                scale_pred[...,2+class_width]=0.5*h_base*np.exp(scale_pred[...,2+class_width])
                scale_pred[...,3+class_width]=0.5*h_base*np.exp(scale_pred[...,3+class_width])
            # scale_pred[...,0+class_width]=2*w_base*scale_pred[...,0+class_width]
            # scale_pred[...,1+class_width]=2*w_base*scale_pred[...,1+class_width]
            # scale_pred[...,2+class_width]=2*h_base*scale_pred[...,2+class_width]
            # scale_pred[...,3+class_width]=2*h_base*scale_pred[...,3+class_width]

            out_preds.append(scale_pred)
        return out_preds


        






'''
    def multi_range_layer(self,x,layer_depth,loop=1,zero_initial=False):
        short_y=self.normal_layer(x,layer_depth,loop=loop,zero_initial=zero_initial)
        long_y=self.dilation_layer(x,layer_depth,loop=loop,dilation_step=2,zero_initial=zero_initial)
        out_y=Concatenate()([short_y,long_y])
        return out_y
    
    def dilation_layer(self,x,layer_depth,loop=1,dilation_step=2,zero_initial=False):#to train confidence, initialize bias to 0
        temp_x=x
        for i in range(loop):
            if zero_initial:
                temp_x=Conv2D(layer_depth,[3,3],padding='same',dilation_rate=dilation_step
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate)
                    ,bias_initializer='zeros')(temp_x)
                temp_x=Conv2D(layer_depth,[3,3],padding='same'
                    ,bias_initializer='zeros'
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_x)
                temp_x=Activation("relu")(temp_x)
            else:
                temp_x=Conv2D(layer_depth,[3,3],padding='same',dilation_rate=dilation_step
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_x)
                temp_x=Conv2D(layer_depth,[3,3],padding='same'
                    ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_x)
                temp_x=Activation("relu")(temp_x)
        return temp_x
    def activation_layer(self,x):
        if self.center_flag:
            confidence_y,box_xy,box_wh=tf.split(x,[class_width,2,2],axis=-1)
            box_xy=tf.sigmoid(box_xy)
            box_wh=tf.tanh(box_wh)
            # confidence_y=tf.sigmoid(confidence_y)
            confidence_y=tf.tanh(confidence_y)
            y=tf.concat([confidence_y,box_xy,box_wh],axis=-1)
        else:
            confidence_y,box_y=tf.split(x,[class_width,4],axis=-1)
            box_y=tf.tanh(box_y)
            confidence_y=tf.sigmoid(confidence_y)
            y=tf.concat([confidence_y,box_y],axis=-1)
        return y
'''