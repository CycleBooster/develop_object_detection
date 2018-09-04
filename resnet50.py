from data import preprocess_input,inverse_input
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications import ResNet50
import cv2
import numpy as np
import json

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

class resnet50():
    def __init__(self):
        self.model=ResNet50(weights='imagenet',include_top=True)
    def pred(self,img_name,type='jpg'):
        img_path = './/test_photo//'+img_name+'.'+type
        x = cv2.imread(img_path,-1)
        x=cv2.resize(x,(224,224))
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        decode_preds,indices_lists=self.decode_predictions(preds)
        print('Predicted:', decode_preds)
    def decode_predictions(self,preds, top=5):
        global CLASS_INDEX
        if len(preds.shape) != 2 or preds.shape[1] != 1000:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 1000)). '
                             'Found array with shape: ' + str(preds.shape))
        if CLASS_INDEX is None:
            fpath = get_file('imagenet_class_index.json',
                             CLASS_INDEX_PATH,
                             cache_subdir='models')
            CLASS_INDEX = json.load(open(fpath))
        results = []
        out_top_indices=[]
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
            results.append(result)
            out_top_indices.append(top_indices)
        return results,out_top_indices
        