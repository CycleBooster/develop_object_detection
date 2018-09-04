import os
import numpy as np
import cv2
import json
from collections import defaultdict

from progress.bar import Bar
from dataset import data_writer,data_reader
from data import preprocess_input,inverse_input

ann_path="D:/coco_dataset/annotations/"
train_list=["instances_train2014.json","instances_valminusminival2014.json"]
validate_list=["instances_minival2014.json"]

image_train_path="D:/coco_dataset/train2014/"
image_val_path="D:/coco_dataset/val2014/"
image_path_dict={"instances_train2014.json":image_train_path,"instances_valminusminival2014.json":image_val_path,"instances_minival2014.json":image_val_path}
class COCO():
    def __init__(self,data_name):
        if data_name=="train_coco":
            self.data_list=train_list
        elif data_name=="val_coco":
            self.data_list=validate_list
        self.data_name=data_name
    def show_data(self):
        for json_data in self.data_list:
            data_path=ann_path+json_data
            image_path=image_path_dict[json_data]
            with open(data_path,'r') as f:
                idToAnn=defaultdict(list)
                image_dict={}
                cat_dict={}
                data_set=json.load(f)
                images=data_set["images"]
                annotations=data_set["annotations"]
                categorys=data_set["categories"]
                for image in images:
                    image_dict[image["id"]]=image
                for ann in annotations:
                    idToAnn[ann["image_id"]].append(ann)
                for cat in categorys:
                    cat_dict[cat["id"]]=cat["name"]
                print(cat_dict)
                id_list=[id for id in image_dict]
                id_list.sort()
                for id in id_list:
                    image=image_dict[id]
                    img_name=image["file_name"]
                    img = cv2.imread(image_path + img_name,1)

                    ann_list=idToAnn[id]
                    
                    for ann in ann_list:
                        class_name=cat_dict[ann["category_id"]]
                        x,y,width,height=ann["bbox"]
                        x_max,y_max=x+width,y+height
                        x,y,x_max,y_max=(int)(x),(int)(y),(int)(x_max),(int)(y_max)
                        cv2.rectangle(img,(x,y),(x_max,y_max),(255,255,255),2)
                        cv2.putText(img,class_name,(x,y),0, 1,(255,255,255),1)
                    cv2.imshow('x',img)
                    cv2.waitKey()
    def build_dataset(self,end_name):
        label_path="./data/hdf5/"+self.data_name+end_name+"_label.npy"
        img_shape=(384,384)
        cv_img_size=(img_shape[1],img_shape[0])
        writer=data_writer(self.data_name+end_name)
        writer.build_dataset("input",img_shape+(3,))
        labels=[]
        for json_data in self.data_list:
            data_path=ann_path+json_data
            image_path=image_path_dict[json_data]
            with open(data_path,'r') as f:
                print("start decode",data_path)
                idToAnn=defaultdict(list)
                image_dict={}
                cat_dict={}
                data_set=json.load(f)
                images=data_set["images"]
                annotations=data_set["annotations"]
                categorys=data_set["categories"]
                for image in images:
                    image_dict[image["id"]]=image
                for ann in annotations:
                    idToAnn[ann["image_id"]].append(ann)
                for cat in categorys:
                    cat_dict[cat["id"]]=cat["name"]
                final_cat_list=[cat for index, cat in cat_dict.items()]
                name2class={name:(int)(i) for i,name in enumerate(final_cat_list)}
                id_list=[id for id in image_dict]
                id_list.sort()
                list_len=len(id_list)
                bar=Bar('Processing', max=list_len,fill='-')
                debug_index=0
                for id in id_list:
                    debug_index=debug_index+1
                    writer.add_size()
                    image=image_dict[id]
                    img_name=image["file_name"]
                    img = cv2.imread(image_path + img_name,1)

                    img_height,img_width=img.shape[0],img.shape[1]
                    img=cv2.resize(img,cv_img_size)
                    input=preprocess_input(img)#inverse order of channel
                    writer.write("input",input)

                    ann_list=idToAnn[id]
                    obj_list=[]
                    for ann in ann_list:
                        class_name=cat_dict[ann["category_id"]]
                        x,y,width,height=ann["bbox"]
                        x_mid,y_mid=(x+width/2)/img_width,(y+height/2)/img_height
                        w,h = width/img_width,height/img_height
                        obj_list.append([w,h,x_mid,y_mid,name2class[class_name]])

                    labels.append(obj_list)
                    bar.next()
                bar.finish()
                    

        labels=np.array(labels)
        np.save(label_path,labels)

def build_COCO(data_name,end_name):
    coco_data=COCO(data_name)
    coco_data.build_dataset(end_name)

if __name__ == '__main__':
    coco_data=COCO("val_coco")
    coco_data.show_data()