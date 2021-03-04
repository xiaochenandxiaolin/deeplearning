import tensorflow as tf
from config import image_path
from data import img_data,get_label,get_model
import numpy as np
import time as t
import config
import os



if __name__ == '__main__':
    # GPU settings
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    #get image dataset
    # t1=t.time()
    img = img_data(image_path)

    t1=t.time()
    predict=[]
    #load model
    # for i in range (0,5):
    model=get_model(1)
    t2=t.time()
    for j in range(len(img)):
        predictions=model(img[j])
        classification = np.argmax(predictions)
        # print(classification)
        predict.append(classification)
        # print(t.time()-t2)
        # print(t2)
    print(predict)
    fine_label=get_label(max(predict, key=predict.count))
    print(fine_label)
    print(t.time()-t2)