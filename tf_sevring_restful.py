import requests as requests
import numpy as np
import tensorflow as tf
from config import image_path
from data import get_label,get_model,img_data
import numpy as np
import time as t
import config
import os
import json
# from tfsevringlin import img_data
def url(num):
    if num==0:
        SERVER_URL1 = 'http://218.17.70.226:8501/v1/models/model1:predict'
        return SERVER_URL1
    elif num==1:
        SERVER_URL2 = 'http://218.17.70.226:8501/v1/models/model2:predict'
        return SERVER_URL2
    elif num==2:
        SERVER_URL3 = 'http://218.17.70.226:8501/v1/models/model3:predict'
        return SERVER_URL3
    elif num==3:
        SERVER_URL4 = 'http://218.17.70.226:8501/v1/models/model4:predict'
        return SERVER_URL4
    elif num==4:
        SERVER_URL5 = 'http://218.17.70.226:8501/v1/models/model5:predict'
        return SERVER_URL5
def prediction():
    t1=t.time()
    predict=[]
    img=img_data(image_path) 
    # print(img[1])
    print(t.time()-t1)
    t2=t.time()
    for a in range(0,1):
        SERVER_URL=url(a)
        for i in range(0,5):
            j=json.dumps(img[i].numpy().tolist())
            predict_request='{"instances":%s}' % j
            # print(t.time()-t1)
        # print(predict_request)
            start_time=t.time()
            response = requests.post(SERVER_URL, data=predict_request)
            print(t.time()-start_time)
        # print(type(response))
            # t2=t.time()
            prediction = response.json()['predictions'][0]
            # print(t.time()-t2)
            classification = np.argmax(prediction)
            # print(t.time()-t2)
            predict.append(classification)
            # print(t.time()-t2)
        print(predict)
        fine_label=get_label(max(predict, key=predict.count))
    print(t.time()-t1,t.time()-t2)
    print(fine_label)
    
if __name__ == "__main__":
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    prediction()