import tensorflow as tf
from config import save_model_dir,image_path
from train import get_model
from data import img_data
import numpy as np
import time as t
import json

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    t1=t.time()

    # load the model
    # model = get_model()
    # model.load_weights(save_model_dir)
    model = tf.saved_model.load(save_model_dir)

    #image data
    predict=[]
    probabilit=[]
    img=img_data(image_path)


    #Take the category with the most predicted results

    # for i in range(len(img)):
    #     image=tf.expand_dims(img[i],0)
    #     predictions = model(image, training=False)
    #     probabilities=tf.nn.softmax(predictions)
    #     label = np.argmax(predictions,1)
    #     probability=np.max(tf.keras.backend.eval((tf.math.top_k(probabilities,1)).values))
    #     predict.append(int(label))
    # print(max(predict, key=predict.count))
    # print('time:',(t.time()-t1))


    # Choose the most likely

    for i in range(len(img)):
        image=tf.expand_dims(img[i],0)
        predictions = model(image, training=False)
        probabilities=tf.nn.softmax(predictions)
        label = np.argmax(predictions,1)
        probability=np.max(tf.keras.backend.eval((tf.math.top_k(probabilities,1)).values))
        probabilit.append(float(probability))
        predict.append(int(label))
    # print(predict[probabilit.index(max(probabilit))])
    print('time:',(t.time()-t1))

    # get json file
    a=str(predict[probabilit.index(max(probabilit))])
    h =open('class.json',encoding='utf-8')  
    js=json.load(h)
    lb=js[a]
    print(lb)