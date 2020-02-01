import os
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 导入一些具体的工具
from pokemon import  load_pokemon, normalize, denormalize
from resnet import ResNet                   # 导入模型

# 预处理的函数，复制过来。
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [244, 244])

    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y

batchsz = 8

# creat train db   一般训练的时候需要shuffle。其它是不需要的。
images, labels, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))  # 变成个Dataset对象。
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz) # map函数图片路径变为内容。
# crate validation db
images2, labels2, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# create test db
images3, labels3, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)


# 训练样本太小了，resnet网络表达能力很强。这里换成4层小的网络了。
resnet = keras.Sequential([
    layers.Conv2D(16,5,3),
    layers.MaxPool2D(3,3),
    layers.ReLU(),
    layers.Conv2D(64,5,3),
    layers.MaxPool2D(2,2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(64),
    layers.ReLU(),
    layers.Dense(5)
])


# 首先创建Resnet18
# resnet = ResNet(5)
resnet.build(input_shape=(batchsz, 224, 224, 3))
resnet.summary()

# monitor监听器, 连续5个验证准确率不增加，这个事情触发。
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=20

)

# 网络的装配。
resnet.compile(optimizer=optimizers.Adam(lr=1e-4),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# 完成标准的train，val, test;
# 标准的逻辑必须通过db_val挑选模型的参数，就需要提供一个earlystopping技术，
resnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=1000,
           callbacks=[early_stopping])   # 1个epoch验证1次。触发了这个事情，提前停止了。
resnet.evaluate(db_test)













