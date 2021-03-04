from __future__ import absolute_import, division, print_function
import tensorflow as tf
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_every_n_epoch,dataset_dir
from data import generate_datasets, load_and_preprocess_image
import math
from nets import se_resnet,vgg19,efficientnet,resnext,alexnet
import os
import config

def get_model(num):
    if num==0:
        return se_resnet.se_resnet_101()
    elif num==1:
        return resnext.resnext()
    elif num==2:
        return vgg19.VGG19()
    elif num==3:
        return efficientnet.efficient_net()
    elif num==4:
        return alexnet.AlexNet()

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

# def get_model():
#         return se_resnet.se_resnet_101()


def print_model_summary(network):
    # input_shape=tf.keras.Input(shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),name='images')
    # x=tf.keras.backend.placeholder(shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),name='images')
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


if __name__ == '__main__':
    # GPU settings
    
    # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    # tf.data.Dataset.cache(dataset_dir)
    # for i in range(0,5):
    i=0
    save_model_dir=save_model(i)
# get the dataset
    # train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    model = get_model(i)
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function 
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                    EPOCHS,
                                                                                    step,
                                                                                    math.ceil(train_count / BATCH_SIZE),
                                                                                    train_loss.result().numpy(),
                                                                                    train_accuracy.result().numpy()))

        for features in valid_dataset:
            valid_images, valid_labels = process_features(features)
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
            "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                EPOCHS,
                                                                train_loss.result().numpy(),
                                                                train_accuracy.result().numpy(),
                                                                valid_loss.result().numpy(),
                                                                valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if (epoch + 1) % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    # save weights
    # model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    tf.saved_model.save(model, save_model_dir)

    tf.compat.v1.reset_default_graph()
    # convert to tensorflow lite format
    # converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)