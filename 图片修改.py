import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def remove_foreground_and_fill(image_path, mask_path, crop_size):
    # 读取图像和掩模
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # 假设掩模是灰度图

    # 确保掩模是二值化的
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 随机生成裁剪区域的起始点
    h, w = mask.shape
    start_x = np.random.randint(0, w - crop_size[1] + 1)
    start_y = np.random.randint(0, h - crop_size[0] + 1)

    # 裁剪掩模和图像
    cropped_mask = mask[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]
    cropped_image = image[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]

    # 使用inpaint方法进行前景消除并填补
    result = cv2.inpaint(cropped_image, cropped_mask, 3, cv2.INPAINT_TELEA)

    # 将填补后的图像放回原图的相应位置
    result_image = image.copy()
    result_image[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]] = result

    return result_image


def process_image(img_name, img_folder, label_folder, save_dir):
    if img_name.lower().endswith('.jpg'):
        label_name = img_name[:-4] + '_bin.png'  # 构造标签文件名
        if label_name in os.listdir(label_folder):  # 确保标签文件存在
            # 调用remove_foreground_and_fill函数处理图像和掩模
            result_image = remove_foreground_and_fill(
                os.path.join(img_folder, img_name),
                os.path.join(label_folder, label_name),
                (500, 1710)  # 定义裁剪尺寸为500x1710像素
            )

            # 保存处理后的图像到save_files文件夹
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, result_image)


def write_img_label_txt(base_dir, dataset_type):
    # 创建保存txt文件的目录
    save_dir = os.path.join(base_dir, 'save_files')
    os.makedirs(save_dir, exist_ok=True)

    # 获取img和label文件夹的路径
    img_folder = os.path.join(base_dir, dataset_type, 'img')
    label_folder = os.path.join(base_dir, dataset_type, 'label')

    img_names = [img_name for img_name in os.listdir(img_folder) if img_name.lower().endswith('.jpg')]

    with ThreadPoolExecutor() as executor:
        futures = []
        for img_name in img_names:
            futures.append(executor.submit(process_image, img_name, img_folder, label_folder, save_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            future.result()


# 基础目录
base_dir = r'E:\git\unet_seg\unet\original_data\dataset_A'

# 处理test和train文件夹
for dataset_type in ['train']:
    write_img_label_txt(base_dir, dataset_type)

print('All images have been processed and saved.')
