'''
@Description: 
@Author: Ren Qian
@Date: 2019-10-17 09:50:42
'''
import glob
import random

# 更改为自己数据集的存放路径，指向training里的image_2
image_path = '/home/rq/data/kitti_object/image_2/training/image_2/'
# 以下是各个txt文件的生成路径
train_val_file = '/home/rq/data/kitti_object/train_val.txt'
train_file = '/home/rq/data/kitti_object/train.txt'
val_file = '/home/rq/data/kitti_object/val.txt'

# 遍历所有图像
all_image = []
for jpg_file in glob.glob(image_path + '*.png'):
    all_image.append(jpg_file)
print('一共有' + str(len(all_image)) + '张图片')

with open(train_val_file, 'w') as tf:
    for image in all_image:
        tf.write(image + '\n')

# 随机取一半，作为训练数据
train_image = random.sample(all_image, int(len(all_image)*0.7))
print('训练用' + str(len(train_image)) + '张图片')

with open(train_file, 'w') as tf:
    for image in train_image:
        tf.write(image + '\n')

# 另一半作为验证数据
val_image = []
for image in all_image:
    if image not in train_image:
        val_image.append(image)
print('验证用' + str(len(val_image)) + '张图片')

with open(val_file, 'w') as tf:
    for image in val_image:
        tf.write(image + '\n')