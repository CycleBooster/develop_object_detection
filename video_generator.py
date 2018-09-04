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
class data_generator():
    def __init__(self,data_name,type="video",data_type="mp4",final_size=None):
        self.data_name=data_name
        self.type=type
        self.data_type=data_type
        self.frame_size=(224,224)
        self.queue= Queue()
    def __get_cap(self):
        video_path="./data/video/"+self.data_name+"."+self.data_type
        if self.type=="video":
            self.cap = cv2.VideoCapture(video_path)
        elif self.type=="camera":
            self.cap = cv2.VideoCapture(0)
        else:
            print("error type")
            return -1
        if cap.isOpened()== False:
            print("error in opening data")
            return -1
    def setsize(self,size):
        self.frame_size=size
    def data_process(self):
        error=self.__get_cap()
        if error==-1:
            self.cap.release()
            return
        buffer_multiple=10
        while 1:
            end, frame = cap.read()
            if end:
                break
            while self.queue.qsize()>=buffer_multiple:
                sleep(0.01)
            img=cv2.resize(frame,self.frame_size)
            input=preprocess_input(img)#inverse order of channel
            self.queue.put(input)
        self.cap.release()
    def generator(self):
        self.p=Process(target=self.data_process,daemon=True)
        self.p.start()
        while True:
            if self.queue.empty()==False:
                yield self.queue.get()


