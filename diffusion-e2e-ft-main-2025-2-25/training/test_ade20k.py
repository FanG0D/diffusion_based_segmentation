import torch
import matplotlib.pyplot as plt
from dataloaders.load import ADE20K
import numpy as np

def test_ade20k_dataset():
    
    dataset = ADE20K(root_dir="/mnt/diff_based_seg/train_dataset/ade20k", transform=False)
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    
    print("Sample keys:", sample.keys())
    
    print("\nTensor shapes:")
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Seg shape: {sample['seg'].shape}")
    print(f"Annotation shape: {sample['annotation'].shape}")
    
    print("\nValue ranges:")
    print(f"RGB range: [{sample['rgb'].min():.2f}, {sample['rgb'].max():.2f}]")
    print(f"Seg range: [{sample['seg'].min():.2f}, {sample['seg'].max():.2f}]")
    print(f"Annotation range: [{sample['annotation'].min():.2f}, {sample['annotation'].max():.2f}]")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    rgb_img = (sample['rgb'].numpy() + 1) / 2  
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    
    plt.subplot(132)
    seg_img = (sample['seg'].numpy() + 1) / 2  
    seg_img = np.transpose(seg_img, (1, 2, 0))
    plt.imshow(seg_img)
    plt.title('Segmentation')
    
    plt.subplot(133)
    ann_img = sample['annotation'].numpy()
    plt.imshow(ann_img[0], cmap='tab20')
    plt.title('Annotation')

    plt.savefig('ade20k_test.png')
    plt.close()

if __name__ == "__main__":
    test_ade20k_dataset()