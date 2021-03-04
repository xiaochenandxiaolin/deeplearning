import tensorflow as tf
# from config import save_model_dir
from data import generate_datasets
from train import get_model, process_features
import config
import time
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

if __name__ == '__main__':
    t1=time.time()
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    for i in range(0,5):
        save_model_dir=save_model(i)
        # get the original_dataset
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
        # load the model
        # model = get_model()
        # model.load_weights(filepath=save_model_dir)
        model = tf.saved_model.load(save_model_dir)

        # Get the accuracy on the test set
        loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        @tf.function
        def test_step(images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            test_accuracy(labels, predictions)

        for features in test_dataset:
            test_images, test_labels = process_features(features)
            test_step(test_images, test_labels)
            print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                            test_accuracy.result()))

        print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))
