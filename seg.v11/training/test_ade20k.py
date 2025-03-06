import torch
import matplotlib.pyplot as plt
from dataloaders.load import ADE20K
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ade20k_dataset():
    """测试ADE20K数据集加载器"""
    try:
        # 初始化数据集
        dataset = ADE20K(root_dir="/mnt/diff_based_seg/train_dataset/ade20k/ade20k_train", transform=True)
        logger.info(f"Dataset size: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        logger.info("Sample keys: %s", sample.keys())
        
        # 检查张量形状
        logger.info("\nTensor shapes:")
        expected_keys = ['rgb', 'seg', 'annotation', 'depth']
        for key in expected_keys:
            if key in sample:
                logger.info(f"{key} shape: {sample[key].shape}")
        
        # 检查值范围
        logger.info("\nValue ranges:")
        for key in expected_keys:
            if key in sample:
                tensor = sample[key]
                logger.info(f"{key} range: [{tensor.min():.2f}, {tensor.max():.2f}]")
        
        # 可视化
        plt.figure(figsize=(20, 5))
        
        # RGB图像
        plt.subplot(141)
        rgb_img = (sample['rgb'].numpy() + 1) / 2
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        plt.imshow(rgb_img)
        plt.title('RGB Image')
        plt.axis('off')
        
        # 分割图(如果存在)
        plt.subplot(142)
        if 'seg' in sample:
            seg_img = (sample['seg'].numpy() + 1) / 2
            seg_img = np.transpose(seg_img, (1, 2, 0))
            plt.imshow(seg_img)
        plt.title('Segmentation (RGB)')
        plt.axis('off')
        
        # 深度图(如果存在)
        plt.subplot(143)
        if 'depth' in sample:
            depth_img = (sample['depth'].numpy() + 1) / 2
            # 如果深度图是3通道的，只取第一个通道
            if depth_img.shape[0] == 3:
                depth_img = depth_img[0]  # 只取第一个通道
            elif depth_img.ndim == 3:
                depth_img = depth_img.mean(axis=0)  # 如果是RGB格式，取平均值
            plt.imshow(depth_img, cmap='gray')
            logger.info(f"Depth image shape after processing: {depth_img.shape}")
        plt.title('Depth Map')
        plt.axis('off')
        
        # 语义分割标注
        plt.subplot(144)
        ann_img = sample['annotation'].numpy()
        plt.imshow(ann_img, cmap='tab20')
        plt.title('Annotation')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('ade20k_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("测试图像已保存为 'ade20k_test.png'")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_ade20k_dataset()