import os
import numpy as np
from keras import backend as K
from lxml import objectify
import cv2
from dataset import data_writer,data_reader
from progress.bar import Bar
import progressbar
import matplotlib.pyplot as plt
import random
from progress import bar
def preprocess_input(x, dim_ordering='default',is_list=False):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    
    x=x.astype('float')
    if is_list:
        if dim_ordering == 'th':
            x[:,0, :, :] -= 103.939
            x[:,1, :, :] -= 116.779
            x[:,2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[::-1, :, :]
            assert x.shape[1]==3
        else:
            x[:,:, :, 0] -= 103.939
            x[:,:, :, 1] -= 116.779
            x[:,:, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            assert x.shape[3]==3
    else:
        if dim_ordering == 'th':
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[::-1, :, :]
            assert x.shape[0]==3
        else:
            x[:, :, 0] -= 103.939
            x[:, :, 1] -= 116.779
            x[:, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            assert x.shape[2]==3
    return x
def inverse_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    x=np.copy(x)
    if dim_ordering == 'th':
        # 'BGR'->'RGB'
        x = x[::-1, :, :]
        x[0, :, :] += 103.939
        x[1, :, :] += 116.779
        x[2, :, :] += 123.68
        assert x.shape[0]==3
    else:
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        assert x.shape[2]==3
    x=x.astype('uint8')
    return x

class2name=['person','bird','cat','cow','dog'
            ,'horse' ,'sheep','aeroplane','bicycle','boat'
            ,'bus' ,'car','motorbike','train','bottle'
            ,'chair','diningtable','pottedplant','sofa','tvmonitor']
# class2name=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_width=len(class2name)
from setup import *
name2class={class2name[i]:i for i in range(class_width)}

def draw_block(photo,size=(7,7)):
    block_h,block_w=photo.shape[0]/size[0],photo.shape[1]/size[1]
    img=np.copy(photo)
    for i in range(size[1]+1):#vertical line
        x=(int)(block_w*i)
        cv2.line(img,(x,0),(x,photo.shape[0]),(0,255,0),1)
    for j in range(size[0]):#horizontal line
        y=(int)(block_h*j)
        cv2.line(img,(0,y),(photo.shape[1],y),(0,255,0),1)
    return img
def build_video():
    load_path='./test_photo/temp_save'
    save_path='./test_photo/test_video.avi'
    save_size=(640,480)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path,fourcc, 20.0, save_size)
    for item in os.listdir(load_path):
        img = cv2.imread(load_path + "/"+item,1)
        img=cv2.resize(img,save_size)
        out.write(img) 
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
def build_video_combine():
    load_path_1='./test_photo/temp_save_center'
    load_path_2='./test_photo/temp_save_connected'
    save_path='./test_photo/test_video_combine.avi'
    cv_save_size=(640,480)
    cv_out_size=(1300,480)
    blank_size=(480,20,3)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path,fourcc, 20.0, cv_out_size)
    for item in os.listdir(load_path_1):
        img_1 = cv2.imread(load_path_1 + "/"+item,1)
        img_1=cv2.resize(img_1,cv_save_size)
        img_2 = cv2.imread(load_path_2 + "/"+item,1)
        img_2=cv2.resize(img_2,cv_save_size)

        blank_img=np.zeros(shape=blank_size,dtype=np.uint8)
        # blank_img.fill(255)
        out_img=np.concatenate([img_1,blank_img,img_2],axis=1)
        out.write(out_img)
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
'''
def get_IOU(answer_box_list,pred_box_list,IOU_thre=0.5):
    IOU_list=[]
    for test_box in pred_box_list:
        test_IOU_list=[]
        for box in answer_box_list:
            if box[1]!=test_box[1]:
                test_IOU_list.append(0)
            else:
                min_box=[min(box[i],test_box[i]) for i in range(2,6)]
                max_box=[max(box[i],test_box[i]) for i in range(2,6)]
                inter_box=[max_box[0],min_box[1],max_box[2],min_box[3]]
                if inter_box[1]-inter_box[0]<0 or inter_box[3]-inter_box[2]<0:
                    inter=0
                else:
                    inter=(inter_box[1]-inter_box[0])*(inter_box[3]-inter_box[2])
                union=(test_box[3]-test_box[2])*(test_box[5]-test_box[4])+(box[3]-box[2])*(box[5]-box[4])-inter
                iou=inter/union
                if iou>=IOU_thre:
                    test_IOU_list.append(iou)
        if len(test_IOU_list)>0:
            max_IOU=max(test_IOU_list)
            if max_IOU>=IOU_thre:
                IOU_list.append(max_IOU)
    # avg_IOU=sum(IOU_list)/len(IOU_list)
    return IOU_list
def statistic_conf(answer_list,pred_list):
    conf_table=np.zeros(shape=(class_width,10))
    bar=Bar('Processing', max=len(answer_list[0]),fill='-')
    for img_index in range(len(answer_list[0])):
        for layer_index in range(len(answer_list)):
            answer_split=np.split(answer_list[layer_index][img_index],[class_width,4+class_width],axis=-1)
            _confidence,_box=answer_split[0],answer_split[1]
            flatten_conf=_confidence.flatten()
            conf_arg=np.flip(np.argsort(flatten_conf),axis=0)
            pred_split=np.split(pred_list[layer_index][img_index],[class_width,4+class_width],axis=-1)
            p_confidence,p_box=pred_split[0],pred_split[1]
            flatten_p_conf=p_confidence.flatten()
            for arg in conf_arg:
                confidence=flatten_conf[arg]
                class_index=(int)(arg%class_width)
                if confidence==1:
                    test_conf=flatten_p_conf[arg]
                    test_out=(int)(test_conf/0.1)
                    if test_out==10:
                        test_out=9
                    conf_table[class_index][test_out]=conf_table[class_index][test_out]+1
                else:
                    break
        bar.next()
    bar.finish()
    number_sum=np.sum(conf_table,axis=-1)
    average_div=np.expand_dims(number_sum,axis=-1)
    average_div=np.where(average_div<=0,np.ones(shape=average_div.shape),average_div)
    conf_table=conf_table/average_div
    return conf_table,number_sum
def exchange_conf(tradition_pred_list,proposed_pred_list):
    out_tradition_list=[]
    out_proposed_list=[]
    for answer_index in range(len(tradition_pred_list)):
        tradition_split=np.split(tradition_pred_list[answer_index],[class_width,4+class_width],axis=-1)
        tradition_confidence,tradition_box=np.copy(tradition_split[0]),np.copy(tradition_split[1])
        proposed_split=np.split(proposed_pred_list[answer_index],[class_width,4+class_width],axis=-1)
        proposed_confidence,proposed_box=np.copy(proposed_split[0]),np.copy(proposed_split[1])
        now_out_tradition=np.concatenate([proposed_confidence,tradition_box],axis=-1)
        now_out_proposed=np.concatenate([tradition_confidence,proposed_box],axis=-1)
        out_tradition_list.append(now_out_tradition)
        out_proposed_list.append(now_out_proposed)
    return out_tradition_list,out_proposed_list
def get_model_pred(name):
    import pickle
    with open('./model/'+name+'.txt', 'rb') as handle:
        preds = pickle.load(handle)
    return preds
def compare_IOU(IOU_thre):
    from data_generator import data_generator
    data_name="validate"
    tradition_pred_list=get_model_pred("tradition")
    # proposed_pred_list=get_model_pred("proposed")
    proposed_pred_list=get_model_pred("connected_real")
    tradition_box_pconf_list,proposed_box_tconf_list=exchange_conf(tradition_pred_list,proposed_pred_list)

    data_gen=data_generator([9],data_name,batch_size=16)
    input_set,answer_list=data_gen.static_data()
    # show_answer_list(answer_list,input_set,center_box_type=False)

    tradition_IOU_list=[]
    proposed_IOU_list=[]
    ex_tradition_IOU_list=[]
    ex_proposed_IOU_list=[]
    bar=Bar('Processing', max=len(input_set),fill='-')
    test_index=-1
    import time
    start_time=time.time()
    print("start test")
    for index,img in enumerate(input_set):
        width,height=img.shape[1],img.shape[0]
        answer_box_list=get_box(answer_list,index,width,height,center_box_type=False,show_thre=1)
        tradition_box_list=get_box(tradition_pred_list,index,width,height,center_box_type=True,show_thre=0.05,flatten=True,take_index=test_index)
        proposed_box_list=get_box(proposed_pred_list,index,width,height,center_box_type=False,show_thre=0.05,flatten=True,take_index=test_index)
        tradition_IOU=get_IOU(answer_box_list,tradition_box_list,IOU_thre=IOU_thre)
        proposed_IOU=get_IOU(answer_box_list,proposed_box_list,IOU_thre=IOU_thre)
        #exchange confidence
        ex_tradition_box_list=get_box(tradition_box_pconf_list,index,width,height,center_box_type=True,show_thre=0.05,flatten=True,take_index=test_index)
        ex_proposed_box_list=get_box(proposed_box_tconf_list,index,width,height,center_box_type=False,show_thre=0.05,flatten=True,take_index=test_index)
        ex_tradition_IOU=get_IOU(answer_box_list,ex_tradition_box_list,IOU_thre=IOU_thre)
        ex_proposed_IOU=get_IOU(answer_box_list,ex_proposed_box_list,IOU_thre=IOU_thre)
        tradition_IOU_list.extend(tradition_IOU)
        proposed_IOU_list.extend(proposed_IOU)
        ex_tradition_IOU_list.extend(ex_tradition_IOU)
        ex_proposed_IOU_list.extend(ex_proposed_IOU)
        bar.next()
    bar.finish()
    avg_tradition_IOU=sum(tradition_IOU_list)/len(tradition_IOU_list)
    avg_proposed_IOU=sum(proposed_IOU_list)/len(proposed_IOU_list)
    avg_ex_tradition_IOU=sum(ex_tradition_IOU_list)/len(ex_tradition_IOU_list)
    avg_ex_proposed_IOU=sum(ex_proposed_IOU_list)/len(ex_proposed_IOU_list)
    print(avg_tradition_IOU)
    print(avg_proposed_IOU)
    print(avg_ex_tradition_IOU)
    print(avg_ex_proposed_IOU)

    print("time=",time.time()-start_time)
    print("end")
'''
def get_box(answer_list,index,width,height,center_box_type,show_thre=conf_thre,color_list=None,color_start_index=None,flatten=True,take_index=-1):
    box_list=[]
    for answer_index in range(len(answer_list)):
        if take_index!=-1 and take_index!=answer_index:
            continue
        temp_box_list=[]
        grid_y,grid_x,grid_depth=answer_list[answer_index].shape[1],answer_list[answer_index].shape[2],answer_list[answer_index].shape[3]
        answer_split=np.split(answer_list[answer_index][index],[1,5,5+class_width],axis=-1)
        _confidence,_box,_cat=answer_split[0],answer_split[1],answer_split[2]
        flatten_conf=_confidence.flatten()
        conf_arg=np.flip(np.argsort(flatten_conf),axis=0)
        block_h,block_w=height/grid_y,width/grid_x
        for arg in conf_arg:
            confidence=flatten_conf[arg]
            if confidence>=show_thre:
                # class_index=(int)(arg%class_width)
                # arg=(arg-class_index)/class_width
                k=(int)(arg%grid_depth)
                arg=(arg-k)/grid_depth
                j=(int)(arg%grid_x)
                arg=(arg-j)/grid_x
                i=(int)(arg%grid_y)
                x_base,y_base=block_w*(j+0.5),block_h*(i+0.5)
                box=_box[i][j][k]
                if center_box_type:
                    x,y,w,h=box
                    x_mid,y_mid=block_w*(j+x),block_h*(i+y)
                    left,right=x_mid-w/2,x_mid+w/2
                    top,bottom=y_mid-h/2,y_mid+h/2
                else:
                    left,right,top,bottom=box
                    left,right=x_base-left,x_base+right
                    top,bottom=y_base-top,y_base+bottom

                if color_list!=None and color_start_index!=None:
                    color_index=(k+color_start_index[answer_index])%len(color_list)
                    color=color_list[color_index]
                else:
                    color=(255,255,255)
                if flatten:
                    box_list.append([confidence,class_index,left,right,top,bottom,color,x_base,y_base])
                else:
                    temp_box_list.append([confidence,class_index,left,right,top,bottom,color,x_base,y_base])
            else:
                break
        if not flatten:
            box_list.append(temp_box_list)
    return box_list
def draw_box(box,image,width,height,out_width,out_height):
    conf,x,x_max,y,y_max,color,x_base,y_base=box[0],box[2],box[3],box[4],box[5],box[6],box[7],box[8]
    x,x_max=(int)(x/width*out_width),(int)(x_max/width*out_width)
    y,y_max=(int)(y/height*out_height),(int)(y_max/height*out_height)
    x_base,y_base=(int)(x_base/width*out_width),(int)(y_base/height*out_height)
    class_name=class2name[box[1]]
    x_mid,y_mid=(int)((x+x_max)/2),(int)((y+y_max)/2)
    cv2.rectangle(image,(x,y),(x_max,y_max),color,4)
    cv2.circle(image,(x_mid,y_mid), 5, (0,255,0), -1)
    conf_out="%.2f"%(conf)
    cv2.putText(image,class_name+conf_out,(x,y),0, 1,color,3)
    # cv2.putText(image,conf_out,(x_base+10,y_base),0, 1,(255,255,255),1)#used for debug
    # cv2.circle(image,(x_base,y_base), 5, (0,0,255), -1)#used for debug
    return image
def show_answer_list(answer_list,inputs,name="",file_save=False,show=True,wait=True,delete_by_IOU=None,out_size=(500,500),flatten=True,center_box_type=is_center):
    save_path='./test_photo/temp_save'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if len(inputs)!=len(answer_list[0]):
        print("error in data length",len(inputs),len(answer_list[0]))
        return
    grid_depth_list=[answer.shape[3] for answer in answer_list]
    color_start_index=[sum(grid_depth_list[:i]) for i in range(len(grid_depth_list))]
    color_list=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128),(0,128,128)]
    #blue, green, red, 青色 cyan, 品紅色 magenta, 黃色 yellow, 鳧綠Teal, 紫紅Purple, 橄欖色Olive
    import time
    for index,img in enumerate(inputs):
        time_list=[]
        time_list.append(time.time())#debug
        img=inverse_input(img)
        out_img=cv2.resize(img,out_size)
        width,height=img.shape[1],img.shape[0]
        out_width,out_height=out_size
        time_list.append(time.time())#debug
        box_list=get_box(answer_list,index,width,height,center_box_type,conf_thre,color_list,color_start_index,flatten=flatten)
        time_list.append(time.time())#debug
        concat_list=[]
        if delete_by_IOU!=None and not flatten:
            print("should flatten if delete_by_IOU")
        if delete_by_IOU!=None and flatten:
            box_list=IOU_box_delete_draw(box_list,delete_by_IOU)
            time_list.append(time.time())#debug
        if flatten:
            for box in box_list:
                out_img=draw_box(box,out_img,width,height,out_width,out_height)
        else:
            for layer_index,layer_box_list in enumerate(box_list):
                width_count,height_count=answer_list[layer_index].shape[2],answer_list[layer_index].shape[1]
                temp_img=np.copy(out_img)
                temp_img=draw_block(temp_img,(height_count,width_count))#used for debug
                for box in layer_box_list:
                    temp_img=draw_box(box,temp_img,width,height,out_width,out_height)
                concat_list.append(temp_img)
            out_img=cv2.hconcat(concat_list)
            # out_img=np.concatenate(concat_list,axis=1) 
        time_list.append(time.time())#debug
        if file_save:
            cv2.imwrite(save_path+'/'+name+'.jpg',out_img)
        if show:
            cv2.imshow('x',out_img)
            if wait:
                cv2.waitKey()
            else:
                cv2.waitKey(25)
        time_list.append(time.time())#debug
        # for i in range(len(time_list)-1):
        #     time_cost=time_list[i+1]-time_list[i]
        #     print(time_cost,end=" ")
        # print()

def show_IOU_list(pred_list,answer_list,inputs,name="",file_save=False,show=True,delete_by_IOU=None,out_size=(500,500),center_box_type=is_center):
    if len(inputs)!=len(answer_list[0]):
        print("error in data length")
        return
    grid_depth_list=[answer.shape[3] for answer in answer_list]
    color_start_index=[sum(grid_depth_list[:i]) for i in range(len(grid_depth_list))]
    color_list=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128),(0,128,128)]
    #blue, green, red, 青色 cyan, 品紅色 magenta, 黃色 yellow, 鳧綠Teal, 紫紅Purple, 橄欖色Olive
    for index,img in enumerate(inputs):
        img=inverse_input(img)
        out_img=cv2.resize(img,out_size)
        width,height=img.shape[1],img.shape[0]
        out_width,out_height=out_size
        answer_box_list=get_box(answer_list,index,width,height,center_box_type)
        pred_box_list=get_box(pred_list,index,width,height,center_box_type,conf_thre,color_list,color_start_index)
        if delete_by_IOU!=None:
            answer_box_list=IOU_box_delete_draw(answer_box_list,delete_by_IOU)
            pred_box_list=IOU_box_delete_draw(pred_box_list,delete_by_IOU)
        for box_list in [answer_box_list,pred_box_list]:
            for box in box_list:
                out_img=draw_box(box,out_img,width,height,out_width,out_height)
        if file_save:
            cv2.imwrite('./test_photo/yolo_out'+name+'.jpg',out_img)
        if show:
            cv2.imshow('x',out_img)
            cv2.waitKey()
def IOU_box_delete_draw(box_list,IOU_thre):
    final_box_list=[]
    while(len(box_list)>0):
        conf_list=np.array([i[0] for i in box_list])
        conf_max_index=np.argmax(conf_list)
        now_box=box_list[conf_max_index]
        box_list.pop(conf_max_index)
        neighbor_index_list=[]
        for index,box in enumerate(box_list):
            min_box=[min(box[i],now_box[i]) for i in range(2,6)]
            max_box=[max(box[i],now_box[i]) for i in range(2,6)]
            inter_box=[max_box[0],min_box[1],max_box[2],min_box[3]]
            if inter_box[1]-inter_box[0]<0 or inter_box[3]-inter_box[2]<0:
                inter=0
            else:
                inter=(inter_box[1]-inter_box[0])*(inter_box[3]-inter_box[2])
            union=(now_box[3]-now_box[2])*(now_box[5]-now_box[4])+(box[3]-box[2])*(box[5]-box[4])-inter
            iou=inter/union
            if iou>IOU_thre:
                neighbor_index_list.append(index)
        combine_list=[box for i,box in enumerate(box_list) if i in neighbor_index_list]
        combine_list.append(now_box)
        box_list=[box for i,box in enumerate(box_list) if i not in neighbor_index_list]#remove box in combine_list
        # total_conf=sum([box[0] for box in combine_list])
        total_sum=sum([1 for box in combine_list])
        final_box_end=[ sum([box[i]/total_sum  for box in combine_list]) for i in range(2,6)]
        final_box=[now_box[0]]
        clas_statistics=np.zeros((class_width))
        for box in combine_list:
            clas_statistics[box[1]]=clas_statistics[box[1]]+1
        clas=np.argmax(clas_statistics)
        final_box.append(clas)
        final_box.extend(final_box_end)
        # final_box.append((0,0,0))
        final_box.append((255,255,255))
        final_box.extend([now_box[7],now_box[8]])
        final_box_list.append(final_box)#conf,class,x_start,x_end,y_start,y_end,color,x_base,y_base
    return final_box_list
def cluster(box_base,data_set_path,grid_list,reload=False,base_grid=7,up_bound=7):
    if os.path.isfile(data_set_path+'/wh_data.npy') and reload==False:
        wh_data=np.load(data_set_path+'/wh_data.npy')
    else:
        wh_data=[]
        list_len=len([path for path in os.listdir(data_set_path+'/annotations')])
        bar = progressbar.ProgressBar(max_value=list_len)
        print("data_len=",list_len)
        i=0
        for path in os.listdir(data_set_path+'/annotations'):
            with open(data_set_path + '/Annotations/'+path,'r') as f:
                xml = f.read().replace('\n', '')
                annotation=objectify.fromstring(xml)
                width,height=annotation.size.width,annotation.size.height
                for obj in annotation.object:
                    b = obj.bndbox
                    w,h = (b.xmax - b.xmin)/width,(b.ymax-b.ymin)/height
                    wh_data.append([w,h])
            bar.update(i)
            i=i+1
        wh_data=np.array(wh_data)
        np.save(data_set_path+'/wh_data.npy',wh_data)
    multi_scale_data=[]
    for grid_size in grid_list:
        for data in wh_data:
            w,h=data
            w,h=w*grid_size,h*grid_size
            if w<=up_bound and h<=up_bound:
                multi_scale_data.append([w,h])
    print("start k-means")
    stop_threshold=len(multi_scale_data)/200
    print("stop_threshold=",stop_threshold)
    done_array=np.zeros(shape=len(multi_scale_data))
    color=['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_size=len(color)
    plt.ion()
    loop=0
    while True:
        cluster_list=[]
        done=0
        for i in range(len(box_base)):
            cluster_list.append([])
        for i,data in enumerate(multi_scale_data):
            w,h=data
            IOU_list=[]
            for index in range(len(box_base)):
                box_w,box_h=box_base[index][0],box_base[index][1]
                inter_w,inter_h=min(box_w,w),min(box_h,h)
                inter=inter_w*inter_h
                union=box_w*box_h+w*h-inter
                IOU=inter/union/(index*0.1+1)
                IOU_list.append(IOU)
            cluster_index=np.argmax(IOU_list)
            best_IOU=IOU_list[cluster_index]
            cluster_list[cluster_index].append([w,h,best_IOU])
            if done_array[i]!=cluster_index:
                done=done+1
                done_array[i]=cluster_index

        for i,data in enumerate(cluster_list):
            w,h=[item[0] for item in data],[item[1] for item in data]
            w=np.squeeze(w)
            h=np.squeeze(h)
            plt.plot(w,h,color[i%color_size]+'o')
        plt.pause(0.01)
        
        for i in range(len(box_base)):
            # weight_sum=sum([ data[2] for data in cluster_list[i]])
            # box_base[i][0],box_base[i][1]=sum([ data[0]*data[2]/weight_sum for data in cluster_list[i]]),sum([ data[1]*data[2]/weight_sum for data in cluster_list[i]])
            weight_sum=len(cluster_list[i])
            box_base[i][0],box_base[i][1]=sum([ data[0]/weight_sum for data in cluster_list[i]]),sum([ data[1]/weight_sum for data in cluster_list[i]])
        
        loop=loop+1
        if done<stop_threshold:
            break
    print("loop=",loop)
    print("box_base=",box_base)
    plt.ioff()
    plt.savefig(data_set_path+'/cluster'+str(len(box_base))+'.jpg')
    plt.show()
def draw_cluster(box_base,data_set_path,IOU_ratio=0.5):
    if os.path.isfile(data_set_path+'/wh_data.npy'):
        wh_data=np.load(data_set_path+'/wh_data.npy')
        print()
    else:
        wh_data=[]
        list_len=len([path for path in os.listdir(data_set_path+'/annotations')])
        bar = progressbar.ProgressBar(max_value=list_len)
        print("data_len=",list_len)
        i=0
        for path in os.listdir(data_set_path+'/annotations'):
            with open(data_set_path + '/Annotations/'+path,'r') as f:
                xml = f.read().replace('\n', '')
                annotation=objectify.fromstring(xml)
                width,height=annotation.size.width,annotation.size.height
                grid_w,grid_h=width/7,height/7
                for obj in annotation.object:
                    b = obj.bndbox
                    w,h = (b.xmax - b.xmin)/grid_w,(b.ymax-b.ymin)/grid_h
                    wh_data.append([w,h])
            bar.update(i)
            i=i+1
        wh_data=np.array(wh_data)
        np.save(data_set_path+'/wh_data.npy',wh_data)
    color=['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_size=len(color)
    cluster_list=[]
    for i in range(len(box_base)+1):
        cluster_list.append([])
    for i,data in enumerate(wh_data):
        w,h=data
        IOU_list=[]
        for index in range(len(box_base)):
            box_w,box_h=box_base[index][0],box_base[index][1]
            inter_w,inter_h=min(box_w,w),min(box_h,h)
            inter=inter_w*inter_h
            union=box_w*box_h+w*h-inter
            IOU=inter/union
            IOU_list.append(IOU)
        cluster_index=np.argmax(IOU_list)
        best_IOU=IOU_list[cluster_index]
        if best_IOU<IOU_ratio:
            cluster_list[len(box_base)].append([w,h])
        else:
            cluster_list[cluster_index].append([w,h])

    for i,data in enumerate(cluster_list):
        w,h=[item[0] for item in data],[item[1] for item in data]
        w=np.squeeze(w)
        h=np.squeeze(h)
        plt.plot(w,h,color[i%color_size]+'o')
    plt.savefig(data_set_path+'/cluster_draw.jpg')
    plt.show()
def answer_statistics_list(pred_list,file_path,end_name):
    color=['b', 'g', 'r','k']
    color_size=len(color)
    color_interval=1/color_size
    step=pred_list[0].shape[0]
    for layer_index,layer in enumerate(pred_list):
        grid_y,grid_x,grid_depth=layer.shape[1],layer.shape[2],layer.shape[3]
        conf_cluster=[]
        for i in range(color_size):
            conf_cluster.append([])
        for index in range(step):
            conf_true,conf_diff,box_w,box_h=np.split(layer[index],4,axis=-1)
            # print(conf_true.shape)
            for i in range(grid_y):
                for j in range(grid_x):
                    k=np.argmin(conf_diff[i][j])
                    if conf_true[i][j][k][0]==1:
                        interval=(int)(conf_diff[i][j][k][0]/color_interval)
                        if interval==color_size:
                            interval=color_size-1
                        if interval>color_size:
                            print(conf_diff[i][j][k][0],"   ",interval)
                        conf_cluster[interval].append([box_w[i][j][k][0],box_h[i][j][k][0]])
        for i,data in enumerate(conf_cluster):
            w,h=[item[0] for item in data],[item[1] for item in data]
            w=np.squeeze(w)
            h=np.squeeze(h)
            plt.plot(w,h,color[i%color_size]+'o')
        plt.savefig(file_path+str(layer_index)+'_'+end_name+'.jpg')
        plt.clf()


'''
def statistics_AP(pred_list,answer_list,test_size):
    if len(pred_list[0])!=len(answer_list[0]):
        print("error in data length")
        return
    total_category_AP_list=[]
    for i in range(class_width):
        total_category_AP_list.append([])
    height,width=test_size
    IOU_thre=0.5
    # bar = progressbar.ProgressBar(max_value=len(answer_list[0]))
    for index in range(len(answer_list[0])):
        answer_box_list=get_box(answer_list,index,width,height,show_thre=1)
        answer_box_list=IOU_box_delete_draw(answer_box_list,0.95)
        pred_box_list=get_box(pred_list,index,width,height,show_thre=0.05)
        # import time
        # start_time=time.time()
        # print("start test")
        category_AP_list=get_Ap(pred_box_list,answer_box_list,IOU_thre)
        # end_time=time.time()
        # print("time=",end_time-start_time)
        for i in range(class_width):
            total_category_AP_list[i].extend(category_AP_list[i])

        # bar.update(index+1)
    total_AP_list=[]#AP in each category
    for i in range(class_width):
        ap_list=total_category_AP_list[i]
        if len(ap_list)==0:
            total_AP_list.append(-1)
        else:
            total_AP_list.append(sum(ap_list)/len(ap_list))
            print(i,ap_list)
    temp_total_AP_list=[ap for ap in total_AP_list if ap!=-1]
    mAP=sum(temp_total_AP_list)/len(temp_total_AP_list)
    print("mAP=",mAP)
def get_Ap(pred_box_list,answer_box_list,IOU_thre):
    category_AP_list=[]
    for category in range(class_width):
        box_conf=np.array([box[0] for box in pred_box_list if box[1]==category])
        if box_conf.shape[0]==0:
            category_AP_list.append([])
            continue
        arg_sort=np.argsort(box_conf)
        arg_sort=arg_sort[::-1]
        test_box_list=[pred_box_list[index] for index in arg_sort]
        IOU_list=[]
        #get IOU information
        for test_box in test_box_list:
            test_IOU_list=[]
            for box in answer_box_list:
                if box[1]!=test_box[1]:
                    test_IOU_list.append(0)
                else:
                    min_box=[min(box[i],test_box[i]) for i in range(2,6)]
                    max_box=[max(box[i],test_box[i]) for i in range(2,6)]
                    inter_box=[max_box[0],min_box[1],max_box[2],min_box[3]]
                    if inter_box[1]-inter_box[0]<0 or inter_box[3]-inter_box[2]<0:
                        inter=0
                    else:
                        inter=(inter_box[1]-inter_box[0])*(inter_box[3]-inter_box[2])
                    union=(test_box[3]-test_box[2])*(test_box[5]-test_box[4])+(box[3]-box[2])*(box[5]-box[4])-inter
                    iou=inter/union
                    test_IOU_list.append(iou>=IOU_thre)
            IOU_list.append(test_IOU_list)
        #get AP
        np_IOU_list=np.array(IOU_list)
        max_IOU_list=np.max(np_IOU_list,axis=1)
        case_list=np.split(np_IOU_list,np_IOU_list.shape[1],axis=1)
        finish_index_list=[]
        for case in case_list:
            test_index_list=[index for index,value in enumerate(case) if value==True]
            if len(test_index_list)>0:
                finish_index=min(test_index_list)
            else:
                finish_index=-1
            finish_index_list.append(finish_index)
        finish_index_list.sort()
        precision_list=[sum(max_IOU_list[:index+1])/(index+1)for index in range(len(max_IOU_list))]
        recall_precision_list=[]
        end_index=finish_index_list[-1]
        for list_index,finish_index in enumerate(finish_index_list):
            if finish_index==-1:
                continue
            recall=(list_index+1)/len(finish_index_list)
            precision=max(precision_list[finish_index:end_index+1])
            recall_precision_list.append([recall,precision])
        recall_precision_list=np.array(recall_precision_list)
        # print(recall_precision_list)
        test_precision_sum=0
        for test_recall in np.arange(0,1.1,0.1):
            cut_list=[item for item in recall_precision_list if item[0]>=test_recall]
            if len(cut_list)>0:
                test_precision_sum=test_precision_sum+cut_list[0][1]
        ap=test_precision_sum/11
        category_AP_list.append([ap])
    return category_AP_list
def answer_statistics(preds,answers,file_path,end_name):
    if len(preds)!=len(answers):
        print("error in data length")
        return
    # plt.ion()
    color=['b', 'g', 'r','k']
    color_size=len(color)
    color_interval=1/color_size
    conf_cluster=[]
    grid_y,grid_x,grid_depth=answers.shape[1],answers.shape[2],answers.shape[3]
    for i in range(color_size):
        conf_cluster.append([])
    for index,answer in enumerate(answers):
        answer_split=np.split(answers[index],[1,1+class_width,5+class_width],axis=-1)
        _confidence,_class,_box=answer_split[0],answer_split[1],answer_split[2]
        pred_split=np.split(preds[index],[1,1+class_width,5+class_width],axis=-1)
        p_confidence,p_class,p_box=pred_split[0],pred_split[1],pred_split[2]
        p_confidence=preds[index][...,0]
        for i in range(grid_y):
            for j in range(grid_x):
                k=np.argmax(p_confidence[i][j])
                if _confidence[i][j][k][0]>conf_thre:
                    box=_box[i][j][k]
                    w,h = box[0],box[1]
                    conf_diff=_confidence[i][j][k][0]-p_confidence[i][j][k]
                    interval=(int)(conf_diff/color_interval)
                    if interval==color_size:
                        interval=color_size-1
                    if interval>color_size:
                        print(conf_diff,"   ",interval)
                    conf_cluster[interval].append([w,h])
    for i,data in enumerate(conf_cluster):
        w,h=[item[0] for item in data],[item[1] for item in data]
        w=np.squeeze(w)
        h=np.squeeze(h)
        plt.plot(w,h,color[i%color_size]+'o')
    plt.savefig(file_path+end_name+'.jpg')
def combine_build_datasets(data_name,end_name="",max_grid_depth=5,model_input_size=(224,224,)):
    writer=data_writer(data_name+end_name)
    writer.build_dataset("input",model_input_size+(3,))
    for i in range(max_grid_depth):
        writer.build_dataset("answer"+str(i+1),(model_input_size[0]/32,model_input_size[1]/32,i+1,5+class_width))
    dataset_list=[]
    if data_name=="train7_total":
        dataset_list.append("./data/VOC2007")
        dataset_list.append("./data/VOC2012")
    elif data_name=="train7":
        dataset_list.append("./data/VOC2007")
    elif data_name=="validate":
        dataset_list.append("./data/VOCvalidate")
    else:
        print("error in data name")
    list_len=0
    for path in dataset_list:
        list_len=list_len+len([path for path in os.listdir(path+'/annotations')])
    print(list_len)
    bar=Bar('Processing', max=list_len,fill='-', suffix='%(percent)d%%')
    for path in dataset_list:
        build_datasets(path,writer,bar,max_grid_depth,model_input_size)
    bar.finish()
def build_datasets(data_set_path,writer,bar,max_grid_depth=5,model_input_size=(224,224,)):
    for path in os.listdir(data_set_path+'/annotations'):
        with open(data_set_path + '/Annotations/'+path,'r') as f:
            writer.add_size()
            xml = f.read().replace('\n', '')
            annotation=objectify.fromstring(xml)
            img = cv2.imread(data_set_path + '/JPEGImages/' + annotation.filename,-1)
            origin_size=img.shape
            width,height,=origin_size[1],origin_size[0]
            grid_y,grid_x=(int)(model_input_size[0]/32),(int)(model_input_size[1]/32)
            block_h,block_w=height/grid_y,width/grid_x
            img=cv2.resize(img,model_input_size)
            img=preprocess_input(img)#inverse order of channel
            input=np.copy(img)
            writer.write("input",input)
            for i in range(max_grid_depth):
                confidence=np.zeros((grid_y,grid_x,i+1,1))
                clas=np.zeros((grid_y,grid_x,i+1,class_width))
                box=np.zeros((grid_y,grid_x,i+1,4))
                for obj in annotation.object:
                    b = obj.bndbox
                    x_mid,y_mid=(b.xmax + b.xmin)/2,(b.ymax+b.ymin)/2
                    x_index,y_index=(int)(x_mid/block_w),(int)(y_mid/block_h)
                    if x_index>=grid_x or y_index>=grid_y:
                        print(x_index," ",y_index)
                        print(x_mid," ",y_mid)
                        print(block_w," ",block_h)
                        print(width," ",height)
                    w,h = (b.xmax - b.xmin)/block_w,(b.ymax-b.ymin)/block_h
                    x_diff,y_diff=(x_mid%block_w)/block_w,(y_mid%block_h)/block_h
                    for j in range(i+1):
                        confidence[y_index][x_index][j][0]=1
                        clas[y_index][x_index][j][name2class[obj.name]]=1
                        box[y_index][x_index][j]=np.array([w,h,x_diff,y_diff])
                    answer=np.concatenate((confidence,clas,box),axis=-1)
                writer.write("answer"+str(i+1),answer)
            bar.next()
def read_dataset(data_name,end_name=""):
    reader=data_reader(data_name+end_name)
    data_size=reader.get_size("input")
    print(data_size)
    in_input=reader.get_data("input")
    in_answer=reader.get_data("answer5")
    print(in_input.shape)
    show_answer(in_answer,in_input,file_save=True)
def show_answer(answers,inputs,name="",file_save=False,show=True):
    if len(inputs)!=len(answers):
        print("error in data length")
        return
    grid_y,grid_x,grid_depth=answers.shape[1],answers.shape[2],answers.shape[3]
    color_list=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    for index,img in enumerate(inputs):
        img=inverse_input(img)
        # img=draw_block(img,size=(7,7))
        answer_split=np.split(answers[index],[1,1+class_width,5+class_width],axis=-1)
        _confidence,_class,_box=answer_split[0],answer_split[1],answer_split[2]
        _confidence=answers[index][...,0]
        for i in range(grid_y):
            for j in range(grid_x):
                for k in range(grid_depth):
                    if _confidence[i][j][k]>conf_thre:
                        box=_box[i][j][k]
                        block_h,block_w=img.shape[0]/grid_y,img.shape[1]/grid_x
                        w,h = (int)((box[0])*block_w), (int)((box[1])*block_h)
                        x_base,y_base=block_w*j,block_h*i
                        x,y = (int)(x_base+box[2]*block_w-w/2),(int)(y_base+box[3]*block_h-h/2)
                        x_mid,y_mid=(int)(x_base+box[2]*block_w),(int)(y_base+box[3]*block_h)
                        color=color_list[k]
                        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        cv2.circle(img,(x_mid,y_mid), 5, (0,255,0), -1)
                        class_name=class2name[np.argmax(_class[i][j][k])]
                        conf_out="%.2f"%(_confidence[i][j][k])
                        cv2.putText(img,class_name+conf_out,(x,y),0, 1,color,1)
        if file_save:
            cv2.imwrite('./test_photo/yolo_out'+name+'.jpg',img)
        if show:
            cv2.imshow('x',img)
            cv2.waitKey()
def show_IOU(preds,answers,inputs,file_save=False,save_name="out"):
    if len(inputs)!=len(answers) or len(inputs)!=len(preds):
        print("error in data length",len(inputs),' ',len(answers),' ',len(preds))
    grid_y,grid_x,grid_depth=answers.shape[1],answers.shape[2],answers.shape[3]
    color_list=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    for index,img in enumerate(inputs):
        img=inverse_input(img)
        #img=draw_block(img,size=(7,7))
        answer_split=np.split(answers[index],[1,1+class_width,5+class_width],axis=-1)
        _confidence,_class,_box=answer_split[0],answer_split[1],answer_split[2]
        _confidence=answers[index][...,0]
        pred_split=np.split(preds[index],[1,1+class_width,5+class_width],axis=-1)
        p_confidence,p_class,p_box=pred_split[0],pred_split[1],pred_split[2]
        p_confidence=preds[index][...,0]
        for i in range(grid_y):
            for j in range(grid_x):
                for k in range(grid_depth):
                    color=color_list[k]
                    if _confidence[i][j][k]>conf_thre:
                        box=_box[i][j][k]
                        block_h,block_w=img.shape[0]/grid_y,img.shape[1]/grid_x
                        w,h = (int)((box[0])*block_w), (int)((box[1])*block_h)
                        x_base,y_base=block_w*j,block_h*i
                        x,y = (int)(x_base+box[2]*block_w-w/2),(int)(y_base+box[3]*block_h-h/2)
                        x_mid,y_mid=(int)(x_base+box[2]*block_w),(int)(y_base+box[3]*block_h)
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                        cv2.circle(img,(x_mid,y_mid), 5, (0,255,0), -1)
                        class_name=class2name[np.argmax(_class[i][j][k])]
                        cv2.putText(img,class_name+str(_confidence[i][j][k]),(x,y),0, 1, (0,0,255),1)
                
                    if p_confidence[i][j][k]>conf_thre:
                        box=p_box[i][j][k]
                        block_h,block_w=img.shape[0]/grid_y,img.shape[1]/grid_x
                        w,h = (int)((box[0])*block_w), (int)((box[1])*block_h)
                        x_base,y_base=block_w*j,block_h*i
                        x,y = (int)(x_base+box[2]*block_w-w/2),(int)(y_base+box[3]*block_h-h/2)
                        x_mid,y_mid=(int)(x_base+box[2]*block_w),(int)(y_base+box[3]*block_h)
                        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        cv2.circle(img,(x_mid,y_mid), 5, (128,128,0), -1)
                        class_name=class2name[np.argmax(p_class[i][j][k])]
                        cv2.putText(img,class_name+str(p_confidence[i][j][k]),(x,y),0, 1, (0,255,0),1)
        cv2.imshow('x',img)
        if file_save:
            cv2.imwrite('./test_photo/yolo_'+save_name+str(index)+'.jpg',img)
        cv2.waitKey()
class data_generator():
    def __init__(self,data_name,end_name="_gen",batch_size=8,random_size=False,random_mirror=False,random_width=False,shuffle=False):
        self.final_size_list=[5,7,9]
        self.final_size=max(self.final_size_list)
        self.random_size=random_size
        self.random_mirror=random_mirror
        self.random_width=random_width
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.__data_read(data_name,end_name)
        self.queue= Queue()
    def __data_read(self,data_name,end_name):
        label_path="./data/hdf5/"+data_name+end_name+"_label.npy"
        if os.path.isfile("./data/hdf5/"+data_name+end_name+".hdf5")and os.path.isfile(label_path):
            self.labels=np.load(label_path)
            reader=data_reader(data_name+end_name)
            self.imgs=reader.get_data("input")
        else:
            print("read data")
            dataset_list=[]
            if data_name=="train7_total":
                dataset_list.append("./data/VOC2007")
                dataset_list.append("./data/VOC2012")
            elif data_name=="train7":
                dataset_list.append("./data/VOC2007")
            elif data_name=="validate":
                dataset_list.append("./data/VOCvalidate")
            else:
                print("error in data name")
            list_len=0
            for path in dataset_list:
                list_len=list_len+len([path for path in os.listdir(path+'/annotations')])
            print(list_len)
            bar=Bar('Processing', max=list_len,fill='-', suffix='%(percent)d%%')
            final_size=max(self.final_size_list)
            img_shape=(final_size*32,final_size*32)
            writer=data_writer(data_name+end_name)
            writer.build_dataset("input",img_shape+(3,))
            labels=[]
            print("data image shape",img_shape)
            for path in dataset_list:
                for item in os.listdir(path+'/annotations'):
                    with open(path + '/Annotations/'+item,'r') as f:
                        writer.add_size()
                        xml = f.read().replace('\n', '')
                        annotation=objectify.fromstring(xml)
                        img = cv2.imread(path + '/JPEGImages/' + annotation.filename,-1)
                        width,height=img.shape[1],img.shape[0]
                        img=cv2.resize(img,img_shape)
                        input=preprocess_input(img)#inverse order of channel
                        obj_list=[]
                        for obj in annotation.object:
                            b = obj.bndbox
                            x_mid,y_mid=(b.xmax + b.xmin)/width/2,(b.ymax+b.ymin)/height/2
                            w,h = (b.xmax - b.xmin+1)/width,(b.ymax-b.ymin+1)/height
                            obj_list.append([w,h,x_mid,y_mid,name2class[obj.name]])
                        writer.write("input",input)
                        labels.append(obj_list)
                        bar.next()
            bar.finish()
            self.labels=np.array(labels)
            np.save(label_path,self.labels)
            reader=data_reader(data_name+end_name)
            self.imgs=reader.get_data("input")
    def __random_img_shape(self):
        size_list=list(self.final_size_list)
        size_list.remove(self.final_size)
        list_len=len(size_list)
        self.final_size=size_list[random.randint(0,list_len-1)]
    def __get_data(self,start_index,indexes,buffer_img,buffer_label,grid_depth,layer_count,mirror_flag,width_offset):
        print("get in")
        img_shape=(self.final_size*32,(self.final_size+width_offset)*32)
        __inputs=np.zeros((self.batch_size,)+img_shape+(3,))
        for i in range(self.batch_size):
            img=buffer_img[indexes[i+start_index]]
            if mirror_flag:
                img=cv2.flip(img,1)
            cv_img_shape=((self.final_size+width_offset)*32,self.final_size*32)
            __inputs[i]=cv2.resize(img,cv_img_shape)
        out_list=(__inputs,)
        for i in range(layer_count):
            size=self.final_size*(2**i)
            width_offset=width_offset*(2**i)
            _out=np.zeros((self.batch_size,size,size+width_offset,grid_depth,class_width+5))
            for index in range(self.batch_size):
                label=buffer_label[indexes[index+start_index]]
                _out[index]=self.__set_label(label,grid_depth,size,mirror_flag,width_offset)
            out_list=out_list+(_out,)
        print("before put")
        self.queue.put(out_list)
        # return out_list
    def __set_label(self,label,grid_depth,size,mirror_flag,width_offset):
        __conf=np.zeros((size,size+width_offset,grid_depth,1))
        __clas=np.zeros((size,size+width_offset,grid_depth,class_width))
        __box=np.zeros((size,size+width_offset,grid_depth,4))
        for obj in label:
            y_block_size=1/size
            x_block_size=1/(size+width_offset)
            w,h,x_mid,y_mid,class_num=obj
            if mirror_flag:
                x_mid=1-x_mid
            x_index,y_index=(int)(x_mid/x_block_size),(int)(y_mid/y_block_size)
            x_diff,y_diff=(x_mid-x_index*x_block_size)/x_block_size,(y_mid-y_index*y_block_size)/y_block_size
            w,h=w*(size+width_offset),h*size
            for i in range(grid_depth):
                __conf[y_index][x_index][i][0]=1
                __clas[y_index][x_index][i][class_num]=1
                __box[y_index][x_index][i]=np.array([w,h,x_diff,y_diff])
        __out=np.concatenate([__conf,__clas,__box],axis=-1)
        return __out
    def get_max_batch_index(self):
        data_len=self.labels.shape[0]
        batch_index=(int)(data_len/self.batch_size)
        return batch_index
    def set_final_size(self,final_size):
        self.final_size=final_size
    def debug_test(self):
        print("get in")
    def generator(self,grid_depth=1,layer_count=1):
        buffer_multiple=10
        start_index=0
        data_len=self.labels.shape[0]
        mirror_flag=False
        width_offset=0
        print("WTF??")
        start=True
        while 1:
            if self.random_size==True:
                self.__random_img_shape()
            if self.random_mirror==True:
                mirror_flag=mirror_flag^True
            if self.random_width==True:
                offset_list=[-1,0,1]
                offset_list.remove(width_offset)
                list_len=len(offset_list)
                width_offset=offset_list[random.randint(0,list_len-1)]
            for i in range(buffer_multiple):
                if i==0:
                    img_buffer=self.imgs[start_index:start_index+buffer_multiple*self.batch_size]
                    label_buffer=self.labels[start_index:start_index+buffer_multiple*self.batch_size]
                    index=[i for i in range(len(img_buffer))]
                    if self.shuffle==True:
                        np.random.shuffle(index)
                # yield self.__get_data(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset)
                if start:
                    last_process=Process(target=self.__get_data, args=(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset))
                    # last_process=Process(target=self.debug_test,args=())
                    print("?")
                    last_process.start()
                    print("!")
                    
                    start=False
                else:
                    next_process=Process(target=self.__get_data, args=(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset))
                    print("????")
                    # next_process=Process(target=self.debug_test,args=())
                    next_process.start()
                    last_process.join()
                    yield self.queue.get()
                    last_process=next_process
            
            start_index+=buffer_multiple*self.batch_size
            if start_index+buffer_multiple*self.batch_size > data_len:
                step=(int)((data_len-start_index)/self.batch_size)
                img_buffer=self.imgs[start_index:data_len]
                label_buffer=self.labels[start_index:data_len]
                index=[i for i in range(len(img_buffer))]
                if self.shuffle==True:
                    np.random.shuffle(index)
                for i in range(step):
                    # yield self.__get_data(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset)
                    if start:
                        last_process=Process(target=self.__get_data, args=(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset))
                        last_process.start()
                        start=False
                    else:
                        next_process=Process(target=self.__get_data, args=(i*self.batch_size,index,img_buffer,label_buffer,grid_depth,layer_count,mirror_flag,width_offset))
                        next_process.start()
                        last_process.join()
                        yield self.queue.get()
                        last_process=next_process
                start_index=0
                start=True
                last_process.join()
                yield self.queue.get()




'''