import tensorflow as tf
from config import image_path
from data import img_data
import numpy as np
import time as t
import json
import config
import os
from collections import Counter
def save_model(path_num):
    if path_num==0:
        return config.semodel
    elif path_num==1:
        return config.resxmodel
    elif path_num==2:
        return config.vggmodel
    elif path_num==3:
        return config.effmodel
    elif path_num==4:
        return config.alexmodel
        # get json file
def get_label(string):
    s=str(string)
    h =open('class.json',encoding='utf-8')
    js=json.load(h)
    lb=js[s]
    return lb
def get_model(nums):
    if nums==0:
        model_se = tf.saved_model.load(save_model(0))
        return model_se
    if nums==1:
        model_res = tf.saved_model.load(save_model(1))
        return model_res
    if nums==2:
        model_vgg = tf.saved_model.load(save_model(2))
        return model_vgg
    if nums==3:
        model_eff = tf.saved_model.load(save_model(3))
        return model_eff
    if nums==4:
        model_alex = tf.saved_model.load(save_model(4))
        return model_alex

if __name__ == '__main__':
    # GPU settings
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    #get image dataset
    img = img_data(image_path)
    t1=t.time()
    predict=[]
    #load model
    for i in range (0,5):
        model=get_model(i)
        #get classification
        for j in range(len(img)):
            predictions=model(img[j])
            classification = np.argmax(predictions)
            predict.append(classification)
        print(predict)
        fine_label=get_label(max(predict, key=predict.count))
        print(fine_label)
    print(t.time()-t1)