import os
import random
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm


def trans_255_1(image):
    if len(image.shape) == 2:  # 单通道图像
        mask = image == 255
        image[mask] = 1
    elif len(image.shape) == 3 and image.shape[2] == 3:  # 三通道图像
        white = np.array([255, 255, 255])
        mask = np.all(image == white, axis=-1)
        image[mask] = [1, 1, 1]
    else:
        raise ValueError("Unsupported image format!")
    return image


def resize_image_and_mask(image, mask, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def process_image(image_file, image_dir, mask_dir, output_mask_dir, output_image_dir, target_size):
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file.rsplit('.', 1)[0] + '_bin.png')
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError(f"Image or mask not found for {image_file}")

    resized_image, resized_mask = resize_image_and_mask(image, mask, target_size)
    resized_mask = trans_255_1(resized_mask)

    unique_id = str(uuid.uuid4())
    mask_output_path = os.path.join(output_mask_dir, unique_id + '.png')
    image_output_path = os.path.join(output_image_dir, unique_id + '.jpg')

    cv2.imwrite(mask_output_path, resized_mask)
    cv2.imwrite(image_output_path, resized_image)

    return unique_id


def mask_to_unet(image_dir, mask_dir, output_mask_dir, output_image_dir, train_txt, val_txt, target_size,
                 num_images=2000):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) < num_images:
        raise ValueError(f"Not enough images in directory to sample {num_images} images.")

    # 随机选择 num_images 个文件
    image_files = random.sample(image_files, num_images)

    args = [(image_file, image_dir, mask_dir, output_mask_dir, output_image_dir, target_size) for image_file in
            image_files]

    results = []
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(process_image, *arg) for arg in args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                unique_id = future.result()
                results.append(unique_id)
            except Exception as e:
                print(f"Error processing image: {e}")

    random.shuffle(results)
    split_point = int(len(results) * 0.8)
    train_ids, val_ids = results[:split_point], results[split_point:]

    with open(train_txt, 'a') as train_file:
        for uid in train_ids:
            train_file.write(f"{uid}\n")

    with open(val_txt, 'a') as val_file:
        for uid in val_ids:
            val_file.write(f"{uid}\n")

    print(f"训练集文件名已写入 {train_txt}")
    print(f"验证集文件名已写入 {val_txt}")


if __name__ == "__main__":
    target_size = (1696, 864)
    image_dir = r"E:\git\unet_seg\unet\VOCdevkit\VOC2007\original_data\dataset_A\train\img"
    mask_dir = r"E:\git\unet_seg\unet\VOCdevkit\VOC2007\original_data\dataset_A\train\label"
    output_mask_dir = "SegmentationClass"
    output_image_dir = "JPEGImages"
    output_txt_dir = './ImageSets/Segmentation'

    train_txt = os.path.join(output_txt_dir, 'train.txt')
    val_txt = os.path.join(output_txt_dir, 'val.txt')

    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    mask_to_unet(image_dir, mask_dir, output_mask_dir, output_image_dir, train_txt, val_txt, target_size,
                 num_images=len(os.listdir(image_dir)))