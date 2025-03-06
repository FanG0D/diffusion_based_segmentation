import unittest
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import (
    BaseSegDataset,
    DatasetMode,
    get_dataset,
)

class TestDatasetLoading(unittest.TestCase):
    def setUp(self):
        # 设置测试环境
        self.dataset_config = "C:/Users/Somebody/Desktop/ToBeBetter/code/seg-0305/Diff_based_seg/config/dataset/data_ade20k_val.yaml"
        self.base_data_dir = "C:/Users/Somebody/Desktop/ToBeBetter/code/train_dataset"
        self.cfg_data = OmegaConf.load(self.dataset_config)

    def test_dataset_initialization(self):
        """测试数据集初始化"""
        dataset = get_dataset(
            self.cfg_data, 
            base_data_dir=self.base_data_dir, 
            mode=DatasetMode.RGB_ONLY
        )
        self.assertIsNotNone(dataset, "数据集初始化失败")
        self.assertIsInstance(dataset, BaseSegDataset, "数据集类型不正确")

    def test_dataset_properties(self):
        """测试数据集基本属性"""
        dataset = get_dataset(
            self.cfg_data, 
            base_data_dir=self.base_data_dir, 
            mode=DatasetMode.RGB_ONLY
        )
        self.assertTrue(len(dataset) > 0, "数据集为空")

    def test_dataloader_creation(self):
        """测试DataLoader创建"""
        dataset = get_dataset(
            self.cfg_data, 
            base_data_dir=self.base_data_dir, 
            mode=DatasetMode.RGB_ONLY
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        self.assertIsNotNone(dataloader, "DataLoader创建失败")

    def test_sample_structure(self):
        """测试数据样本结构"""
        dataset = get_dataset(
            self.cfg_data, 
            base_data_dir=self.base_data_dir, 
            mode=DatasetMode.RGB_ONLY
        )
        sample = dataset[0]
        
        # 打印所有键
        print("\n样本包含的所有键:")
        for key in sample.keys():
            print(f"键名: {key}")
            
        # 打印每个键对应数据的形状/类型信息
        print("\n每个键的数据形状/类型:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 类型={type(value)}, 形状={value.shape}")
            elif isinstance(value, (str, int, float)):
                print(f"{key}: 类型={type(value)}, 值={value}")
            else:
                print(f"{key}: 类型={type(value)}")
        
        # 原有的测试代码保持不变
        self.assertIn('rgb_int', sample, "样本缺少rgb_int")
        self.assertIn('rgb_relative_path', sample, "样本缺少rgb_relative_path")
        self.assertIsInstance(sample['rgb_int'], torch.Tensor)
        self.assertEqual(len(sample['rgb_int'].shape), 3, "RGB图像维度不正确")
        self.assertEqual(sample['rgb_int'].shape[0], 3, "RGB通道数不正确")

if __name__ == '__main__':
    unittest.main()