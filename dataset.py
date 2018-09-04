import h5py
import numpy as np
class dataset_writer():
    def __init__(self,name,input_shape,output_shape,write_answer=True):
        self.size=0
        self.in_shape=input_shape
        self.ans_shape=output_shape
        self.data=h5py.File("./data/hdf5/"+name+".hdf5", "w")
        self.write_answer=write_answer
        input_set = self.data.create_dataset("input", (self.size,)+self.in_shape, maxshape=(None,)+self.in_shape)
        if self.write_answer==True:
            answer_set = self.data.create_dataset("answer", (self.size,)+self.ans_shape,maxshape=(None,)+self.ans_shape)
    def write(self,input,output):
        self.size=self.size+1
        input_set=self.data["input"]
        input_set.resize((self.size,)+self.in_shape)
        input_set[self.size-1]=input
        if self.write_answer==True:
            answer_set=self.data["answer"]
            answer_set.resize((self.size,)+self.ans_shape)
            answer_set[self.size-1]=output
class dataset_reader():
    def __init__(self,name,read_answer=True):
        self.data=h5py.File("./data/hdf5/"+name+".hdf5", "r")
        self.read_answer=read_answer
    def get_size(self):
        input_set=self.data.get("input")
        return input_set.shape[0]
    def read_generator(self,batch_size=1):
        input_set=self.data.get("input")
        if self.read_answer==True:
            answer_set=self.data.get("answer")
        size=input_set.shape[0]
        out_len=(int)(size/batch_size)
        while 1:
            for i in range(out_len):
                if self.read_answer==True:
                    yield (input_set[i*batch_size:(i+1)*batch_size],answer_set[i*batch_size:(i+1)*batch_size])
                else:
                    yield input_set[i*batch_size:(i+1)*batch_size]
    def get_data(self):
        input_set=self.data.get("input")
        if self.read_answer==True:
            answer_set=self.data.get("answer")
            return input_set,answer_set
        else:
            return input_set

class data_writer():
    def __init__(self,name):
        self.size=0
        self.data=h5py.File("./data/hdf5/"+name+".hdf5", "w")
    def build_dataset(self,dataset_name,data_shape):
        dataset = self.data.create_dataset(dataset_name, (self.size,)+data_shape, maxshape=(None,)+data_shape)
        # dataset = self.data.create_dataset(dataset_name, (self.size,)+data_shape, maxshape=(None,)+data_shape,compression="lzf")
    def add_size(self):
        self.size=self.size+1
    def write(self,dataset_name,data):
        dataset=self.data[dataset_name]
        dataset.resize(self.size,axis=0)
        dataset[self.size-1]=data
class data_reader():
    def __init__(self,name):
        self.data=h5py.File("./data/hdf5/"+name+".hdf5", "r")
    def get_size(self,dataset_name):
        dataset=self.data.get(dataset_name)
        return dataset.shape[0]
    def get_data(self,dataset_name):
        dataset=self.data.get(dataset_name)
        return dataset