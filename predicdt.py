import itertools
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import cv2
from unet import Unet
from nets.U_ConvAutoencoder import U_ConvAutoencoder
from typing import Tuple, List


# 定义卷积自编码器

class PreCA:
    device: torch.device = None
    model: U_ConvAutoencoder = None
    transform: transforms.Compose = None

    @classmethod
    def initialize_model(cls, u_ca_path: str) -> None:
        # 实例化模型并加载权重
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = U_ConvAutoencoder().to(cls.device)
        cls.model.load_state_dict(torch.load(u_ca_path, map_location=cls.device))
        cls.model.eval()
        # 图像预处理
        cls.transform = transforms.Compose([
            transforms.Resize((1728, 3392)),
            transforms.ToTensor()
        ])

    @classmethod
    def load_image(cls, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        image = cls.transform(image).unsqueeze(0)  # 添加batch维度
        return image.to(cls.device)

    @staticmethod
    def ca_smooth(image: Image.Image) -> Image.Image:
        image_cv2 = np.array(image)
        # 对图像进行闭运算
        closed_image = cv2.morphologyEx(image_cv2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # Step 1: 使用高斯模糊来平滑图像边缘
        blurred = cv2.GaussianBlur(closed_image, (1, 1), 0)
        th = cv2.threshold(blurred, 126, 255, cv2.THRESH_BINARY)[1]

        eroded_image_pil = Image.fromarray(th)
        return eroded_image_pil

    @classmethod
    def infer(cls, image: Image.Image) -> Image.Image:
        image = cls.load_image(image)
        with torch.no_grad():
            output = cls.model(image)
        output = output.squeeze(0).cpu()  # 去除batch维度并移动到CPU
        output_image = transforms.ToPILImage()(output)
        output_image = output_image.resize((3384, 1710), Image.NEAREST)
        return output_image


class PreUnet:

    @staticmethod
    def blend_images_with_colorize(image1: Image.Image, image2: Image.Image, alpha: float = 0.5) -> None:
        red_image1 = ImageOps.colorize(image1.convert("L"), (0, 0, 0), (255, 0, 0))
        green_image2 = ImageOps.colorize(image2.convert("L"), (0, 0, 0), (0, 255, 0))
        blended_image = Image.blend(red_image1, green_image2, alpha)
        blended_image.show()

    @staticmethod
    def calculate_metrics(pred_image: Image.Image, true_image: Image.Image, threshold: int = 1) -> Tuple[int, int, int]:
        pred_gray = pred_image.convert('L')
        true_gray = true_image.convert('L')

        pred_binary = pred_gray.point(lambda x: 0 if x < threshold else 255)
        true_binary = true_gray.point(lambda x: 0 if x < threshold else 255)

        pred_array = np.array(pred_binary)
        true_array = np.array(true_binary)

        # Calculate TP, FP, FN
        TP = np.sum((pred_array == 255) & (true_array == 255))
        FP = np.sum((pred_array == 255) & (true_array == 0))
        FN = np.sum((pred_array == 0) & (true_array == 255))

        return TP, FP, FN

    @staticmethod
    def apply_mask(original_image, mask_imag):
        # 打开原图和mask图片
        original_image = original_image.convert("RGB")
        mask_image = mask_imag.convert("RGB")

        # 获取图片的像素数据
        original_pixels = original_image.load()
        mask_pixels = mask_image.load()

        # 获取图片的尺寸
        width, height = original_image.size

        # 遍历每个像素
        for y in range(height):
            for x in range(width):
                # 如果mask的像素是白色 (255, 255, 255)
                if mask_pixels[x, y] == (255, 255, 255):
                    # 将原图中的对应像素改为绿色 (0, 255, 0)
                    original_pixels[x, y] = (0, 255, 0)

        # 保存结果图片
        return original_image



    @classmethod
    def main(cls, ca_path: str) -> None:
        PreCA.initialize_model(ca_path)
        import os
        from tqdm import tqdm
        ious: List[float] = []
        img_names: List[str] = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                r_image = unet.detect_image(image)
                r_image = PreCA.infer(r_image)  # 自编码器
                r_image = PreCA.ca_smooth(r_image)


                if is_save:
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name.split('.')[0] + '_bin.png'))
                if is_get_iou:
                    label_path = os.path.join(dir_label_path, img_name.split('.')[0] + '_bin.png')
                    label = Image.open(label_path)
                    TP, FP, FN = cls.calculate_metrics(r_image, label)
                    iou = TP / (TP + FP + FN)
                    ious.append(iou)
                    print(f"当前iou{iou}")

                    # cls.blend_images_with_colorize(label, r_image)

        if is_get_iou: print(f"平均iou{np.mean(ious)}")


if __name__ == "__main__":
    name_classes: List[str] = ["background", "lane"]
    dir_origin_path: str = r"E:\git\unet_seg\unet\original_data\dataset_A\test\img"
    # 是否计算IOU，若为True必须填写dir_label_path（label的路径）
    is_get_iou: bool = True
    dir_label_path: str = r"E:\git\unet_seg\unet\original_data\dataset_A\test\Label"
    # 是否保存预测后的图像，若为True必须填写dir_save_path（保存路径的路径）
    is_save: bool = False
    dir_save_path: str = "img_out/"
    # 设置多尺度监督自编码器的权重路径
    u_ca_path: str = 'weights/best_conv_autoencoder1.pth'
    _defaults: dict = {
        "model_path": 'model_data/best80.pth',  # U-Net权重地址
        "num_classes": 2,  # 预测类别算上背景为2
        "backbone": "vgg",
        "input_shape": [1696, 864],  # 图像大小
        "mix_type": 1,
        "cuda": True,  # 是否启用cuda加速
    }
    unet: Unet = Unet(_defaults)
    PreUnet.main(u_ca_path)
