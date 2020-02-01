import os
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')      # 判断tf的版本是否是以‘2.’开头，如果是，则返回True，否则返回False

# 导入一些具体的工具
from pokemon import  load_pokemon, normalize, denormalize
from resnet import ResNet                               # 导入模型

# 预处理的函数，复制过来。
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)             # RGBA
    x = tf.image.resize(x, [256, 256])

    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, max_delta=0.5)    # 在某范围随机调整图片亮度
    x = tf.image.random_contrast(x, 0.1, 0.6)           # 在某范围随机调整图片对比度
    x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y

########################################################################################################################
batchsz = 16

# creat train db  一般训练的时候需要shuffle。其它是不需要的。
images, labels, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))     # 变成个Dataset对象。
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)    # map函数图片路径变为内容。
# crate validation db
images2, labels2, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# create test db
images3, labels3, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)

########################################################################################################################
# 导入别的已经训练好的网络和参数, 这部分工作在keras网络中提供了一些经典的网络以及经典网络训练好的参数。
# 这里使用Vgg19,还把他的权值导入进来。imagenet训练的1000类，我们就把输出层去掉。
net = keras.applications.VGG19(weights='imagenet',
                               include_top=False,
                               pooling='max')

# net.trainable = False                             # 把这部分老的网络，不需要参与反向更新。不训练。为了更好的适应，我下面让2层可以训练;
for i in range(len(net.layers)-4):                  # print(len(model.layers))=23
    net.layers[i].trainable = False

model = keras.Sequential([net, layers.Dense(5)])

model.build(input_shape=(None, 224, 224, 3))
model.summary()

# early_stopping：monitor监听器,当验证集损失值，连续增加小于0时，持续10个epoch，则终止训练。
early_stopping = EarlyStopping(monitor='val_accuracy',
                               min_delta=0.00001,
                               patience=30, verbose=1)

# reduce_lr：当评价指标不在提升时，减少学习率，每次减少10%，当验证损失值，持续3次未减少时，则终止训练。
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01,
                              patience=30, min_lr=0.0000001, verbose=1)

########################################################################################################################
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  # 损失函数

model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=1000,
          initial_epoch=0, callbacks=[early_stopping, reduce_lr])                           # 1个epoch验证1次

model.evaluate(db_test)

