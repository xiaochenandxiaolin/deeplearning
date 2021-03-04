import tensorflow as tf
from config import train_dir, valid_dir, test_dir, train_tfrecord, valid_tfrecord, test_tfrecord
from data import get_images_and_labels
import random


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0.))):
        value = value.numpy()  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    image_paths_and_labels_dict = shuffle_dict(image_paths_and_labels_dict)
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, label in image_paths_and_labels_dict.items():
            print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord)
    dataset_to_tfrecord(dataset_dir=valid_dir, tfrecord_name=valid_tfrecord)
    dataset_to_tfrecord(dataset_dir=test_dir, tfrecord_name=test_tfrecord)