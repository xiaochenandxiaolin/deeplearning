import tensorflow as tf
import pathlib
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord
from read_tfrecord import get_parsed_dataset
import os
import numpy as np
import json
import config

def img_data(path):
    file_name = os.listdir(path)
    # decode
    l=[]
    for i in range(len(file_name)):
        image_name=path+file_name[i]
        image_raws=tf.io.read_file(image_name)
        img_tensor = tf.io.decode_png(contents=image_raws, channels=CHANNELS)
        # resize
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.dtypes.cast(img_tensor, tf.dtypes.float32)
        # image_tenor = img_tensor[np.newaxis,:,:,:]
        # normalization
        image = img_tensor / 255.0
        imag=tf.expand_dims(image,0)
        l.append(imag)
    return l


        # get json file
def get_label(string):
    s=str(string)
    h =open('class.json',encoding='utf-8')
    js=json.load(h)
    lb=js[s]
    return lb

#加载和预处理图像
def load_and_preprocess_image(image):
    # decode
    img_tensor = tf.io.decode_jpeg(contents=image, channels=CHANNELS)
    # resize
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.dtypes.cast(img_tensor, tf.dtypes.float32)
    # normalization
    img = img_tensor / 255.0
    return img

#获取图像和标签
def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((index, label) for label, index in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label

#获取数据集长度
def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count

def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count

