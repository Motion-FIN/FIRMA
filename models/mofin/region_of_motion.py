import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor_image = torch.tensor(image).float().permute(2, 0, 1) / 255.0
    return tensor_image, gray_image

def compute_optical_flow(image0, image1, max_corners=100, num_boxes=5):
    # 텐서를 넘파이 배열로 변환
    image1_ = image0.cpu().numpy().transpose(1, 2, 0)
    image2_ = image1.cpu().numpy().transpose(1, 2, 0)
    # 0과 1 사이의 값으로 스케일링된 이미지를 0부터 255까지의 값으로 변환
    image1_ = (image1_ * 255).astype(np.uint8)
    image2_ = (image2_ * 255).astype(np.uint8)
    # result_frame = image1_.copy()
    # result_frame1 = image2_.copy()
    # plt.imshow(image1_)
    # plt.axis('off')  # Turn off axis
    # plt.show()

    # 이미지를 그레이스케일로 변환
    image1 = cv2.cvtColor(image1_, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2_, cv2.COLOR_BGR2GRAY)

    # 특징점 검출 (Shi-Tomasi 알고리즘)
    corners1 = cv2.goodFeaturesToTrack(image1, max_corners, 0.01, 10)
    if corners1 is None:
        return [], []

    # 옵티컬 플로우 계산 (Lucas-Kanade 알고리즘)
    corners2, status, _ = cv2.calcOpticalFlowPyrLK(image1, image2, corners1, None)

    # 움직임 크기를 기준으로 정렬
    corners_diff = corners2 - corners1
    corners_diff_norm = np.linalg.norm(corners_diff, axis=2)
    sorted_indices = np.argsort(corners_diff_norm.ravel())[::-1]

    # 바운딩 박스 생성 및 정규화된 값 계산
    num_boxes = min(num_boxes, len(sorted_indices))
    normalized_values = []
    bounding_boxes = []
    for i in range(num_boxes):
        index = sorted_indices[i]
        x1, y1 = corners1[index].ravel()
        x2, y2 = corners2[index].ravel()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        x_ = int(x)
        y_ = int(y)
        w_ = int(w)
        h_ = int(h)
        bounding_boxes.append((x_, y_, w_, h_))
        # cv2.rectangle(result_frame1, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)

        if corners_diff_norm.max() == 0:
            normalized_value = 0.0
        else:
            normalized_value = corners_diff_norm[index] / corners_diff_norm.max()
        
        normalized_values.append(normalized_value)
        # plt.imshow(result_frame)
        # plt.axis('off')  # Turn off axis
        # plt.show()
        # plt.imshow(result_frame1)
        # plt.axis('off')  # Turn off axis
        # plt.show()
    return bounding_boxes, normalized_values

def RoMotion(image_path1, image_path2, max_corners=100, num_boxes=5):
    btSize = image_path1.shape[0]
    all_boundingBox = []
    all_normalizedVal = []
    for i in range(btSize):
        bounding_boxes, normalized_values = compute_optical_flow(image_path1[i].cuda(), image_path2[i].cuda(), max_corners, num_boxes)
        all_boundingBox.append(bounding_boxes)
        all_normalizedVal.append(normalized_values)
        # print(f'all_boundingBox:{all_boundingBox}')
    
    outputs = {
        'bounding_box': all_boundingBox
    }
    
    return outputs, all_normalizedVal

# Specify the number of bounding boxes to display
# image_path1 = 'test1.jpg'
# image_path2 = 'test3.jpg'
# max_corners = 300
# num_boxes = 145

# # Draw the bounding boxes on the current frame
# current_frame = cv2.imread(image_path2)
# result_frame = current_frame.copy()

# # 옵티컬 플로우 계산 및 결과 출력
# bounding_boxes, normalized_values = calculate_optical_flow_between_frames(image_path1, image_path2, max_corners, num_boxes)
