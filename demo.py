import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from setup import *

# data_name="train_coco"
# val_name="val_coco"
data_name="train_total"
val_name="validate"
pred_list=['horse_human','dog','chair','human_bicycle']
# pred_list=['dog','dog2','dog3','dog4','cat','cat2','cat3','cat4']
# box_list=[[9.2868186095122809, 10.065021528054263], [7.4429679906540738, 6.866458807908824], [4.1160470766936896, 8.1530679409598221], [3.8602494812012469, 4.744581763059033], [7.1058450210656359, 3.5702718036570853], [1.9813065924675006, 5.3824663128128805], [2.7082649861447456, 2.2513183186595525], [1.1333823591739838, 2.6348957966276521], [0.62970181657903157, 0.96161839281611616]]
# box_list=[[11.65625, 10.1875], [4.875, 6.1875], [3.625, 2.8125], [1.84375, 3.71875], [1.9375, 1.40625], [0.9375, 1.90625], [1.03125, 0.71875], [0.5, 0.9375], [0.3125, 0.40625]]
# box_list=[[2,2], [2,4], [4,2], [2.52,2.52], [2.52, 5.04],[5.04,2.52],[3.2,3.2],[6.4,3.2],[3.2,6.4]]
# box_list=[[box[0]*32,box[1]*32] for box in box_list]
# box_list=[[box[0]*16,box[1]*16] for box in box_list]
# box_list=np.array(box_list,dtype=int)
if __name__ == '__main__':
    from keras_model import ObjectDetector
    test=ObjectDetector("./model/"+data_name+".h5",center_flag=is_center,lr=1e-4,pred_list=pred_list,test_size=(320,320))
    # test.train(data_name=data_name,batch_size=16,epoch=1,keep_train=False)
    # test.pred_print(-1,0,load_weight=True)
    # test.validate(4,0)
    # test.debug_IOU_mask(60)
    
    # test.evaluate(data_name=data_name,test_length=1,test_total=True)
    # test.evaluate(data_name=val_name,index=0,test_length=1,test_total=False)
    # test.evaluate_conf(data_name,index=0,test_length=1,test_total=False)
    # test.test(data_name=data_name,index=0,test_length=60)
    # test.test_AP(data_name=val_name,index=0,test_length=60)
    # test.pred_video("cat_and_dog2",max_size=320,file_save=True)

    # if is_center:
    #     test_kind="tradition"
    # else:
    #     # test_kind="proposed"
    #     test_kind="connected_real"
    # # test.pred_dataset(test_kind)
    # test.test_load_pickle(test_kind)

    # import data
    # # data.build_video()
    # data.build_video_combine()

    # import data
    # data.cluster([[5,5], [3, 5], [5, 3], [3, 3], [1, 1]],"./data/VOC2012",grid_list=[8,10,12],up_bound=10)

    # import data
    # data.draw_cluster([[11.65625, 10.1875], [4.875, 6.1875], [3.625, 2.8125], [1.84375, 3.71875], [1.9375, 1.40625], [0.9375, 1.90625], [1.03125, 0.71875], [0.5, 0.9375], [0.3125, 0.40625]],"./data/VOC2012",IOU_ratio=0.5)
    # box_list=[[11.65625, 10.1875], [4.875, 6.1875], [3.625, 2.8125], [1.84375, 3.71875], [1.9375, 1.40625], [0.9375, 1.90625], [1.03125, 0.71875], [0.5, 0.9375], [0.3125, 0.40625]]

    # from data_generator import data_generator
    # from dataset import data_writer,data_reader
    # from data import *
    # import cv2
    # from time import sleep
    # import data
    # import sys
    # import os
    # gen=data_generator([2],"train_total",final_size=10,random_size=False,batch_size=1,random_mirror=False,random_width=False,random_noise=False)
    # # gen=data_generator([1],val_name,final_size=7,random_size=False,batch_size=1,random_mirror=False,random_width=False)
    # batch_len=gen.get_max_batch_index()
    # print(batch_len)
    # generator=gen.generator()
    # for i in range(20):
    #     input,answer_list=next(generator)
    #     for answer in answer_list:
    #         print(answer.shape)
    #     data.show_answer_list(answer_list,input,flatten=False)

    # from data import compare_IOU
    # compare_IOU(IOU_thre=0)
    
