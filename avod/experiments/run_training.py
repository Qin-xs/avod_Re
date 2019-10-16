'''
@Description: 
@Author: Ren Qian
@Date: 2019-10-11 17:11:25
'''
"""Detection model trainer.

This runs the DetectionModel trainer.
"""

import argparse
import os

import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core import trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def train(model_config, train_config, dataset_config):
    # 读取config文件里面的详细内容，包括：
    # model_config：模型参数
    # train_config：训练参数
    # dataset_config：数据集参数
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    train_val_test = 'train'
    # 包括avod_model 和 rpn_model
    model_name = model_config.model_name

    with tf.Graph().as_default():
        if model_name == 'rpn_model':
            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=dataset)
        elif model_name == 'avod_model':
            model = AvodModel(model_config,
                              train_val_test=train_val_test,
                              dataset=dataset)
        else:
            raise ValueError('Invalid model_name')

        trainer.train(model, train_config)


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    # 从配置文件中读取参数
    default_pipeline_config_path = avod.root_dir() + \
        '/configs/avod_cars_example.config'
    default_data_split = 'train'
    default_device = '1'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path, is_training=True)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    train(model_config, train_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
