
from __future__ import print_function
from __future__ import print_function
# from grpc.beta import implementations
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from data import img_data,get_label
from config import image_path
import time as t
import grpc
import numpy as np
tf.compat.v1.app.flags.DEFINE_string('server', '218.17.70.226:8500',
                            'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS

def model_name(num):
    if num == 0:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'model1'
        request.model_spec.signature_name = 'serving_default'
        return request
    elif num == 1:
        request = predict_pb2.PredictRequest()
        request.model_spec.name='model2'
        request.model_spec.signature_name = 'serving_default'
        return request
    elif num == 2:
        request = predict_pb2.PredictRequest()
        request.model_spec.name='model3'
        request.model_spec.signature_name = 'serving_default'
        return request
    elif num == 3:
        request = predict_pb2.PredictRequest()
        request.model_spec.name='model4'
        request.model_spec.signature_name = 'serving_default'
        return request
    elif num == 4:
        request = predict_pb2.PredictRequest()
        request.model_spec.name='model5'
        request.model_spec.signature_name = 'serving_default'
        return request

def main(_):
    t1=t.time()
    host, port = FLAGS.server.split(':')
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # request = predict_pb2.PredictRequest()
    predict = []
    image_data = img_data(image_path)
    imag_data = []
    for i in range(len(image_data)):
        imag_data.append(image_data[i].numpy().tolist())
    t2=t.time()
    for h in range(0,5):
        request=model_name(h)
        t3=t.time()
        # Build a batch of images. 
        request.inputs['input_1'].CopyFrom(
            tf.make_tensor_proto(imag_data, shape = [len(image_data),224,224,3]))
        result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
        result = result_future.result()
        t4=t.time()
        classes = result.outputs['output_1'].float_val
        prediction = []
        for i in range(0,80):
            prediction.append(classes[i])
            if len(prediction) % 16  ==  0:
                classification = np.argmax(prediction)
                predict.append(classification)
                prediction = []
    print(predict)
    fine_label = get_label(max(predict, key = predict.count))
    print(fine_label,t4-t2,t2-t1,t4-t1)
if __name__  ==  '__main__':
    tf.compat.v1.app.run()