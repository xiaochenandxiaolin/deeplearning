# coding=utf-8
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from data import img_data
from config import image_path
def main():
        # Build a batch of images.
    image_data = img_data(image_path)
    imag_data = []
    for i in range(len(image_data)):
        imag_data.append(image_data[i].numpy().tolist())
    with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'model1'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_1'].CopyFrom(
          tf.make_tensor_proto(imag_data, shape=[len(imag_data),224,224,3]))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main() 