import numpy as np
import torch
from skimage.segmentation import slic
import matplotlib.pyplot as plt

# 가상의 텐서 생성 (16 배치의 이미지, 2 채널, 256x256 크기)
# input_tensor = torch.randn(16, 2, 256, 256)

# 텐서를 이미지로 변환
# input_images = input_tensor.permute(0, 2, 3, 1).cpu().numpy()

def SPSlic_result(images, num_superpixels=800, compactness=10):
    
    input_images = images.permute(0, 2, 3, 1).cpu().numpy()
    # SLIC 알고리즘을 사용하여 슈퍼픽셀 생성
    superpixel_masks = []
    for image in input_images:
        image = (image * 255).astype(np.uint8)  # 이미지 스케일 조정 (0~1 -> 0~255)
        labels = slic(image, n_segments=num_superpixels, compactness=compactness)

        superpixel_masks.append(labels)

    # 슈퍼픽셀 마스크를 텐서로 변환
    superpixel_masks = np.array(superpixel_masks)
    superpixel_masks = torch.from_numpy(superpixel_masks).unsqueeze(1)  # 배치 차원 추가
    
    visualize_superpixel_result(input_images, superpixel_masks)
    print(superpixel_masks.size())  # 결과 출력 (텐서 크기)

    return superpixel_masks

def visualize_superpixel_result(images, masks):
    num_batches = images.shape[0]

    for batch_idx in range(num_batches):
        image = images[batch_idx].transpose(1, 2, 0)  # 채널 순서 변경
        mask = masks[batch_idx, 0]  # 배치 차원 제거

        # 원본 이미지와 슈퍼픽셀 결과를 함께 표시
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Superpixel Result')

        plt.show()

