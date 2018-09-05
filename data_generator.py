import os
import numpy as np
from lxml import objectify
import cv2
from dataset import data_writer,data_reader
from progress.bar import Bar
import random
from multiprocessing import Process,Queue
from data import *
from setup import *
from time import sleep
from decode_coco import build_COCO
class data_generator():
    def __init__(self,grid_depth_list,data_name,end_name="_gen",final_size=None,batch_size=8
    ,random_size=False,random_mirror=False,random_width=False,random_noise=False,med_noise=False,shuffle=False):
        self.data_name=data_name
        self.end_name=end_name
        self.grid_depth_list=grid_depth_list
        self.final_size_list=[6,8,10]
        self.top_ratio=32

        self.final_size=max(self.final_size_list)
        if final_size!=None:
            self.final_size=final_size
        self.random_size=random_size
        self.random_mirror=random_mirror
        self.random_width=random_width
        self.random_noise=random_noise
        self.med_noise=med_noise
        self.set_med_noise=True
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.queue= Queue()
        self.__build_data()
        self.__get_data_len()
    def __build_data(self):
        label_path="./data/hdf5/"+self.data_name+self.end_name+"_label.npy"
        if os.path.isfile("./data/hdf5/"+self.data_name+self.end_name+".hdf5")and os.path.isfile(label_path):
            pass
        else:
            print("build data")
            dataset_list=[]
            if self.data_name=="train_total":
                dataset_list.append("./data/VOC2007")
                dataset_list.append("./data/VOC2012")
            elif self.data_name=="train":
                dataset_list.append("./data/VOC2007")
            elif self.data_name=="validate":
                dataset_list.append("./data/VOCvalidate")
            elif self.data_name=="train_coco" or self.data_name=="val_coco":
                build_COCO(self.data_name,self.end_name)
                return
            else:
                print("error in data name")
            list_len=0
            for path in dataset_list:
                list_len=list_len+len([path for path in os.listdir(path+'/annotations')])
            print(list_len)
            bar=Bar('Processing', max=list_len,fill='-')
            final_size=self.final_size
            img_shape=(final_size*self.top_ratio,final_size*self.top_ratio)
            writer=data_writer(self.data_name+self.end_name)
            writer.build_dataset("input",img_shape+(3,))
            labels=[]
            print("data image shape",img_shape)
            for path in dataset_list:
                for item in os.listdir(path+'/annotations'):
                    with open(path + '/Annotations/'+item,'r') as f:
                        writer.add_size()
                        xml = f.read().replace('\n', '')
                        annotation=objectify.fromstring(xml)
                        img = cv2.imread(path + '/JPEGImages/' + annotation.filename,1)
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
            labels=np.array(labels)
            np.save(label_path,labels)
    def __build_static_data(self,label_name):
        data_path="./data/hdf5/"+label_name+".hdf5"
        if os.path.isfile(data_path):
            pass
        else:
            labels,imgs=self.__data_read()
            self.final_size=(int)(imgs.shape[1]/self.top_ratio)
            writer=data_writer(label_name)
            for index,grid_depth in enumerate(self.grid_depth_list):
                now_size=self.final_size*2**index
                if grid_depth>0:
                    writer.build_dataset("label"+str(index),(now_size,now_size,grid_depth,class_width+5))
            gen_len=self.data_len
            indexes=[i for i in range(labels.shape[0])]
            print(len(indexes))
            print(imgs.shape)
            bar=Bar('Processing', max=gen_len,fill='-')
            for i in range(gen_len):
                writer.add_size()
                answer_list=self.__get_data(1,i,indexes,imgs,labels,False,0,is_static=True)
                data_index=0
                for index,grid_depth in enumerate(self.grid_depth_list):
                    if grid_depth>0:
                        writer.write("label"+str(index),answer_list[data_index])
                        data_index=data_index+1
                bar.next()
            bar.finish()
    def __get_data_len(self):
        label_path="./data/hdf5/"+self.data_name+self.end_name+"_label.npy"
        self.data_len=np.load(label_path).shape[0]
    def __data_read(self):
        label_path="./data/hdf5/"+self.data_name+self.end_name+"_label.npy"
        labels=np.load(label_path)
        reader=data_reader(self.data_name+self.end_name)
        imgs=reader.get_data("input")
        return labels,imgs
    def __random_img_shape(self):
        size_list=list(self.final_size_list)
        size_list.remove(self.final_size)
        list_len=len(size_list)
        self.final_size=size_list[random.randint(0,list_len-1)]
    def __set_med_noise(self,input_img,label,mirror_flag,width_offset):
        noise_mask=np.zeros(shape=input_img.shape)
        noise=np.random.normal(0,50,input_img.shape)
        for obj in label:
            w,h,x_mid,y_mid,class_num=obj
            if mirror_flag:
                x_mid=1-x_mid
            out_w,out_h=w*(self.final_size+width_offset)*self.top_ratio,h*self.final_size*self.top_ratio
            out_x,out_y=x_mid*(self.final_size+width_offset)*self.top_ratio,y_mid*self.final_size*self.top_ratio
            draw_flag=False
            if out_w>2*out_h:
                draw_flag=True
                noise_w=(int)(out_w/20)
                noise_h=out_h
            elif out_h>2*out_w:
                draw_flag=True
                noise_h=(int)(out_h/20)
                noise_w=out_w
            if draw_flag:
                x,x_max=(int)(out_x-noise_w/2),(int)(out_x+noise_w/2)
                y,y_max=(int)(out_y-noise_h/2),(int)(out_y+noise_h/2)
                cv2.rectangle(noise_mask,(x,y),(x_max,y_max),(1,1,1),-1)
        input_img=np.where(noise_mask>=1,noise,input_img)
        return input_img

    def __get_data(self,batch_size,start_index,indexes,buffer_img,buffer_label,mirror_flag,width_offset,is_static=False):
        if is_static==False:
            cv_img_shape=((self.final_size+width_offset)*self.top_ratio,self.final_size*self.top_ratio)
            img_shape=(self.final_size*self.top_ratio,(self.final_size+width_offset)*self.top_ratio)
            _inputs=np.zeros((batch_size,)+img_shape+(3,))
            
            for i in range(batch_size):
                img=buffer_img[indexes[i+start_index]]
                if mirror_flag:
                    img=cv2.flip(img,1)
                tmp_img=cv2.resize(img,cv_img_shape)
                if self.med_noise:
                    label=buffer_label[indexes[i+start_index]]
                    if self.set_med_noise:
                        tmp_img=self.__set_med_noise(tmp_img,label,mirror_flag,width_offset)
                    self.set_med_noise=not self.set_med_noise
                _inputs[i]=tmp_img
            if self.random_noise:
                noise=np.random.normal(0,5,(batch_size,)+img_shape+(3,))
                _inputs=_inputs+noise
            if self.med_noise or self.random_noise:
                max_clip=np.zeros((batch_size,)+img_shape+(3,))
                max_clip.fill(255)
                max_clip=preprocess_input(max_clip,is_list=True)
                min_clip=np.zeros((batch_size,)+img_shape+(3,))
                min_clip=preprocess_input(min_clip,is_list=True)
                _inputs=np.clip(_inputs,min_clip,max_clip)
        answer_list=[]
        for i,grid_depth in enumerate(self.grid_depth_list):
            if grid_depth>0:
                size=self.final_size*(2**i)
                label_width_offset=width_offset*(2**i)
                _out=np.zeros((batch_size,size,size+label_width_offset,grid_depth,class_width+5))
                for index in range(batch_size):
                    label=buffer_label[indexes[index+start_index]]
                    _out[index]=self.__set_label(label,grid_depth,size,mirror_flag,label_width_offset,width_offset)
                answer_list.append(_out)
        if is_static==False:
            return (_inputs,answer_list)
        else:
            return answer_list
    def __set_label(self,label,grid_depth,size,mirror_flag,label_width_offset,width_offset):
        __conf=np.zeros((size,size+label_width_offset,grid_depth,1))
        __cat=np.zeros((size,size+label_width_offset,grid_depth,class_width))
        __box=np.zeros((size,size+label_width_offset,grid_depth,4))
        for obj in label:
            y_block_size=1/size
            x_block_size=1/(size+label_width_offset)
            w,h,x_mid,y_mid,class_num=obj
            if mirror_flag:
                x_mid=1-x_mid
            x_mid_index,y_mid_index=(int)(x_mid/x_block_size),(int)(y_mid/y_block_size)

            if is_center:
                # set center
                out_x_mid,out_y_mid=x_mid/x_block_size-x_mid_index,y_mid/y_block_size-y_mid_index
                out_w,out_h=w*(self.final_size+width_offset)*self.top_ratio,h*self.final_size*self.top_ratio
                for i in range(grid_depth):
                    # test_conf=np.amax(__conf[y_mid_index][x_mid_index][i],axis=-1)
                    test_conf=__conf[y_mid_index][x_mid_index][0]
                    if test_conf>0:
                        continue
                    __conf[y_mid_index][x_mid_index][0]=1
                    temp_cat=np.zeros(class_width)
                    class_num=(int)(class_num)
                    temp_cat[class_num]=1
                    __cat[y_mid_index][x_mid_index][i]=temp_cat
                    __box[y_mid_index][x_mid_index][i]=np.array([out_x_mid,out_y_mid,out_w,out_h])
                    break
            else:
                #set connected grid
                x_start,x_end=x_mid-w/2,x_mid+w/2
                y_start,y_end=y_mid-h/2,y_mid+h/2
                
                grid_x_start_mid,grid_x_end_mid=x_mid_index*x_block_size,(x_mid_index+1)*x_block_size
                grid_y_start_mid,grid_y_end_mid=y_mid_index*y_block_size,(y_mid_index+1)*y_block_size

                center_w,center_h=x_block_size/2,y_block_size/2
                # center_w,center_h=w/8,h/8
                center_x_start,center_x_end=x_mid-center_w/2,x_mid+center_w/2
                center_y_start,center_y_end=y_mid-center_h/2,y_mid+center_h/2
                total_center_area=center_w*center_h
                center_x_start_index,center_x_end_index=(int)(center_x_start/x_block_size),(int)(center_x_end/x_block_size)
                if center_x_end_index>=size+label_width_offset:
                    center_x_end_index=size+label_width_offset-1
                center_y_start_index,center_y_end_index=(int)(center_y_start/y_block_size),(int)(center_y_end/y_block_size)
                if center_y_end_index>=size:
                    center_y_end_index=size-1
                mid_area_x_start=max(grid_x_start_mid,center_x_start)
                mid_area_x_end=min(grid_x_end_mid,center_x_end)
                mid_area_y_start=max(grid_y_start_mid,center_y_start)
                mid_area_y_end=min(grid_y_end_mid,center_y_end)
                max_conf_value=(mid_area_x_end-mid_area_x_start)*(mid_area_y_end-mid_area_y_start)
                for x_index in range(center_x_start_index,center_x_end_index+1):
                    for y_index in range(center_y_start_index,center_y_end_index+1):
                        grid_x_start,grid_x_end=x_index*x_block_size,(x_index+1)*x_block_size
                        grid_y_start,grid_y_end=y_index*y_block_size,(y_index+1)*y_block_size
                        area_x_start=max(grid_x_start,center_x_start)
                        area_x_end=min(grid_x_end,center_x_end)
                        area_y_start=max(grid_y_start,center_y_start)
                        area_y_end=min(grid_y_end,center_y_end)
                        # conf=(area_x_end-area_x_start)*(area_y_end-area_y_start)/total_center_area
                        conf=(area_x_end-area_x_start)*(area_y_end-area_y_start)/max_conf_value

                        grid_x_mid,grid_y_mid=x_index*x_block_size+x_block_size/2 , y_index*y_block_size+y_block_size/2
                        left=(grid_x_mid-x_start)*(self.final_size+width_offset)*self.top_ratio
                        right=(x_end-grid_x_mid)*(self.final_size+width_offset)*self.top_ratio
                        top=(grid_y_mid-y_start)*self.final_size*self.top_ratio
                        bottom=(y_end-grid_y_mid)*self.final_size*self.top_ratio
                        for i in range(grid_depth):
                            # test_conf=np.amax(__conf[y_mid_index][x_mid_index][i],axis=-1)
                            test_conf=__conf[y_mid_index][x_mid_index][0]
                            if test_conf>0:
                                continue
                            __conf[y_mid_index][x_mid_index][0]=1
                            temp_cat=np.zeros(class_width)
                            class_num=(int)(class_num)
                            temp_cat[class_num]=1
                            __cat[y_mid_index][x_mid_index][i]=temp_cat
                            __box[y_mid_index][x_mid_index][i]=np.array([out_x_mid,out_y_mid,out_w,out_h])
                            break
        __out=np.concatenate([__conf,__cat,__box],axis=-1)
        return __out
    def get_max_batch_index(self):
        return (int)(self.data_len/self.batch_size)
    def set_final_size(self,final_size):
        self.final_size=final_size
    def data_process(self):
        labels,imgs=self.__data_read()
        buffer_multiple=10
        start_index=0
        mirror_flag=False
        width_offset=0
        while 1:
            if self.queue.qsize()<3*buffer_multiple:
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
                        img_buffer=imgs[start_index:start_index+buffer_multiple*self.batch_size]
                        label_buffer=labels[start_index:start_index+buffer_multiple*self.batch_size]
                        index=[i for i in range(len(img_buffer))]
                        if self.shuffle==True:
                            np.random.shuffle(index)
                    self.queue.put(self.__get_data(self.batch_size,i*self.batch_size,index,img_buffer,label_buffer,mirror_flag,width_offset))
                
                start_index+=buffer_multiple*self.batch_size
                if start_index+buffer_multiple*self.batch_size > self.data_len:
                    step=(int)((self.data_len-start_index)/self.batch_size)
                    img_buffer=imgs[start_index:self.data_len]
                    label_buffer=labels[start_index:self.data_len]
                    index=[i for i in range(len(img_buffer))]
                    if self.shuffle==True:
                        np.random.shuffle(index)
                    for i in range(step):
                        self.queue.put(self.__get_data(self.batch_size,i*self.batch_size,index,img_buffer,label_buffer,mirror_flag,width_offset))
                    start_index=0
    def generator(self):
        self.p=Process(target=self.data_process,daemon=True)
        self.p.start()
        while True:
            if self.queue.empty()==False:
                yield self.queue.get()
    def test_data(self,start_index=0):
        labels,imgs=self.__data_read()
        index=[i for i in range(len(imgs))]
        now_index=start_index
        while True:
            yield self.__get_data(self.batch_size,now_index,index,imgs,labels,False,0)
            now_index=now_index+self.batch_size
    def static_data(self):
        grid_list_name=""
        for grid_depth in self.grid_depth_list:
            grid_list_name=grid_list_name+str(grid_depth)
        label_name=self.data_name+self.end_name+grid_list_name+"_static_label"
        self.__build_static_data(label_name)
        reader=data_reader(self.data_name+self.end_name)
        input_set=reader.get_data("input")
        reader=data_reader(label_name)
        label_set=[]
        for i,grid_depth in enumerate(self.grid_depth_list):
            if grid_depth>0:
                label_set.append(reader.get_data("label"+str(i)))
        return input_set,label_set
    def analyze_data(self,start_index=0):
        grid_list_name=""
        for grid_depth in self.grid_depth_list:
            grid_list_name=grid_list_name+str(grid_depth)
        label_name=self.data_name+self.end_name+grid_list_name+"_static_label"
        self.__build_static_data(label_name)
        reader=data_reader(self.data_name+self.end_name)
        input_set=reader.get_data("input")
        reader=data_reader(label_name)
        label_set=[]
        for i,grid_depth in enumerate(self.grid_depth_list):
            if grid_depth>0:
                label_set.append(reader.get_data("label"+str(i)))
        fake_input_list=label_set
        fake_input_list.append(input_set)
        return fake_input_list


        # self.p=Process(target=self.data_process,daemon=True)
        # self.p.start()
        # while True:
        #     if self.queue.empty()==False:
        #         inputs,answer_list=self.queue.get()
        #         # print(inputs.shape)
        #         # for answer in answer_list:
        #         #     print(answer.shape)
        #         fake_input_list=answer_list
        #         fake_input_list.append(inputs)
        #         yield (fake_input_list,answer_list)

