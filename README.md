# deep Learning
Classification algorithm based on TensorFlow 2.X.

# characteristic
- The model is saved in savedmodel format for easy deployment

- Use tensorflow_serving+docker to deploy CPU/GPU version (multi-model)

- Including docker start command

- Use the restful/grpc method to send prediction requests, one of which is one at a time in the restful method. grpc is sent in batches, and batch size adaptation does not need to be set.

- Has preheated files

- The preheat file is generated to reduce the delay of starting the model.


# Usage
- tf_sevring_grpc.py(grpc)

- tf_sevring_restful.py----------------------(restful)
- tf_sevring_warmup.py-----------------------(preheated files)
- train.py-----------------------------------(training)
- test.py------------------------------------(local test)
- data.py,dadaset.py-------------------------(utils)
- write_tfrecord.py,read_tfrecord.py---------(tfrecord)
- json_data.py-------------------------------(label)
- config.py----------------------------------(set config)
- evaluate.py--------------------------------(eva)
- test_eva.py--------------------------------(...)


# folder
- nets---------------------------------------(vgg,resnet,...)
- dataset------------------------------------(dataset)
- raw----------------------------------------(Store the original data set)
- test---------------------------------------(Store the picture to be recognized)
- saved_model--------------------------------(save model)
- class.json---------------------------------(label)
- tf_docker.txt------------------------------(docker command)

# Updating
- [ ] Target Detection

# 中文说明
基于tf2.x实现的多种分类算法，可进行多模型协同工作，进行预测，同时包括模型部署需要的文件，实现一个人完成单个分类项目。
# 文件说明
| header |
| ------ |
模型保存为savedmodel格式方便部署
采用tensorflow_serving+docker方式部署CPU/GPU版（多模型）
包括docker启动命令
采用restful/grpc方式发送预测请求，其中restful方式一次一张。grpc按batch发送，batch大小自适应不需要设置。
有预热文件
生成预热文件用于减少启动模型的延时。
存在的问题：restful方式耗时长,grpc方式图片转数组耗时较长。 json文件生成不稳定
文件功能说明：
tf_sevring_grpc.py-------------------------grpc方式
tf_sevring_restful.py----------------------restful方式
tf_sevring_warmup.py-----------------------预热文件生成
train.py-----------------------------------模型训练
test.py------------------------------------本地循环加载模型预测
data.py,dadaset.py-------------------------一些函数
write_tfrecord.py,read_tfrecord.py---------数据集操作
json_data.py-------------------------------标签文件生成
config.py----------------------------------参数设置
evaluate.py--------------------------------测试集测试
test_eva.py--------------------------------单一模型针对可能性最高，出现次数最多两种方式的预测
blender------------------------------------存放blender脚本相关数据
nets---------------------------------------存放模型网络
dataset------------------------------------存放分割后的数据集和tfrecord文件
raw----------------------------------------存放原始数据集
test---------------------------------------存放待识别图片
saved_model--------------------------------存放训练后的模型
class.json---------------------------------标签信息
tf_docker.txt------------------------------启动docker命令及一些参数信息
