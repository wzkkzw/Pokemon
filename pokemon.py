import  os, glob
import  random, csv
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_csv(root, filename, name2label):
    """ 加载CSV文件！
    :param root:            root:数据集根目录
    :param filename:        filename:csv文件名
    :param name2label:      name2label:类别名编码表
    :return:
    """
    # 判断.csv文件是否已经存在！
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        # 1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0  图片路径和标签!
                writer.writerow([img, label])
            print('written into csv file:', filename)

    # read from csv file
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels

# 加载pokemon数据集的工具！
def load_pokemon(root, mode='train'):
    """ 加载pokemon数据集的工具！
    :param root:    数据集存储的目录
    :param mode:    mode:当前加载的数据是train,val,还是test
    :return:
    """
    # 创建数字编码表,范围0-4;
    name2label = {}  # "sq...":0   类别名:类标签;  字典 可以看一下目录,一共有5个文件夹,5个类别：0-4范围;
    for name in sorted(os.listdir(os.path.join(root))):     # 列出所有目录;
        if not os.path.isdir(os.path.join(root, name)):#os.path.isdir用于判断某一对象(需提供绝对路径)是否为目录
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息;保存索引文件images.csv
    # [file1,file2,], 对应的标签[3,1] 2个一一对应的list对象。
    # 根据目录,把每个照片的路径提取出来,以及每个照片路径所对应的类别都存储起来，存储到CSV文件中。
    images, labels = load_csv(root, 'images.csv', name2label)

    # 图片切割成，训练70%，验证15%，测试15%。
    if mode == 'train':                                                     # 70% 训练集
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':                                                     # 15% = 70%->85%  验证集
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:                                                                   # 15% = 70%->85%  测试集
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label

# 数据normalize
# 下面这2个值均值和方差，怎么得到的。其实是统计所有imagenet的图片(几百万张)的均值和方差；
# 所有者2个数据比较有意义，因为本质上所有图片的分布都和imagenet图片的分布基本一致。
# 这6个数据基本是通用的，网上一搜就能查到。
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x, mean=img_mean, std=img_std):
    # x shape: [224, 224, 3]
    # mean：shape为1；这里用到了广播机制。我们安装好右边对齐的原则，可以得到如下；
    # mean : [1, 1, 3], std: [3]        先插入1
    # mean : [224, 224, 3], std: [3]    再变为224
    x = (x - mean)/std
    return x

# 数据normalize之后，这里有一个反normalizaion操作。比如数据可视化的时候，需要反过来。
def denormalize(x, mean=img_mean, std=img_std):
    x = x * std + mean
    return x

def preprocess(x,y):
    # x: 图片的路径，
    # y：图片的数字编码
    x = tf.io.read_file(x)                  # 通过图片路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # RGBA 这里注意有些图片不止3个通道。还有A,透明通道。
    x = tf.image.resize(x, [244, 244])      # 图片重置的，这里224*224,刚好resnet大小匹配的，方便查看。

    # data augmentation, 0~255    首先做一个数据增强！这个操作必须在normalizaion之前(因为是针对图片的。)
    # x = tf.image.random_flip_up_down(x)   # 随机的做一个上和下的翻转。如果全都翻转，相当于没有增加。随机选择一部分翻转。
    x= tf.image.random_flip_left_right(x)   # 随机的做一个左和右的翻转。
    # x = tf.image.random_crop(x, [224, 224, 3]) # 图片裁剪，这里注意这里裁剪到224*224，所以resize不能是224，比如250,250不然什么也没做。

    # x: [0,255]=> 0~1 或者-0.5~0.5   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1) 调用函数；
    x = normalize(x)

    y = tf.convert_to_tensor(y)

    return x, y

def main():
    import  time
    images, labels, table = load_pokemon('/Users/wzk/PythonCode/MyPy01/宝可梦/pokeman', 'train')
    # 图片的路径
    print('images', len(images), images)
    # 图片的标签
    print('labels', len(labels), labels)
    # 编码表，所对应的类别名字。
    print(table)
    # 数据集装载
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    # 数据集预处理tensorboard --logdir logs
    db = db.shuffle(1000).map(preprocess).batch(32)
    # 我们做一个可视化，图片可视化出来。
    #终端命令:tensorboard --logdir logs
    writter = tf.summary.create_file_writer('logs')

    for step, (x, y) in enumerate(db):
        # 这里x的大小: [32, 224, 224, 3]
        # 这里y: [32]
        with writter.as_default():
            #可视化的时候需要将标准化的图片反标准化
            # x = denormalize(x)
            tf.summary.image('img', x, step=step, max_outputs=9)  # 一次记录9张图片。
            time.sleep(5)                                         # 如果显示感觉太快，每5秒刷新一次batch。

if __name__ == '__main__':
    main()