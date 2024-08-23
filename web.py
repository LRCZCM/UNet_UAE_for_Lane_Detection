import os
import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
from PIL.Image import Image
from torchvision import transforms
from PIL import Image
from unet import Unet
from nets.U_ConvAutoencoder import U_ConvAutoencoder
from typing import Tuple, List

# Constants and configuration
DEFAULTS = {
    "model_path": 'model_data/8414_8376.pth',
    "num_classes": 2,
    "backbone": "vgg",
    "input_shape": [1696, 864],
    "mix_type": 1,
    "cuda": torch.cuda.is_available(),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((1728, 3392)),
    transforms.ToTensor()
])


class PreCA:
    model: U_ConvAutoencoder = None

    @classmethod
    def initialize_model(cls, u_ca_path: str) -> None:
        cls.model = U_ConvAutoencoder().to(DEVICE)
        cls.model.load_state_dict(torch.load(u_ca_path, map_location=DEVICE))
        cls.model.eval()

    @classmethod
    def unload_model(cls) -> None:
        cls.model = None
        torch.cuda.empty_cache()

    @classmethod
    def load_image(cls, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        image = TRANSFORM(image).unsqueeze(0)
        return image.to(DEVICE)

    @staticmethod
    def ca_smooth(image: Image.Image) -> Image.Image:
        image_cv2 = np.array(image)
        closed_image = cv2.morphologyEx(image_cv2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        blurred = cv2.GaussianBlur(closed_image, (1, 1), 0)
        th = cv2.threshold(blurred, 126, 255, cv2.THRESH_BINARY)[1]
        return Image.fromarray(th)

    @classmethod
    def infer(cls, image: Image.Image) -> Image.Image:
        image_tensor = cls.load_image(image)
        with torch.no_grad():
            output = cls.model(image_tensor)
        output = output.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output)
        return output_image.resize((3384, 1710), Image.NEAREST)


class PreUnet:

    @staticmethod
    def calculate_metrics(pred_image: Image.Image, true_image: Image.Image, threshold: int = 1) -> Tuple[int, int, int]:
        pred_binary = pred_image.convert('L').point(lambda x: 0 if x < threshold else 255)
        true_binary = true_image.convert('L').point(lambda x: 0 if x < threshold else 255)

        pred_array = np.array(pred_binary)
        true_array = np.array(true_binary)

        TP = np.sum((pred_array == 255) & (true_array == 255))
        FP = np.sum((pred_array == 255) & (true_array == 0))
        FN = np.sum((pred_array == 0) & (true_array == 255))

        return TP, FP, FN

    @staticmethod
    def apply_mask(original_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        original_image = original_image.convert("RGB").resize((3384, 1710), Image.NEAREST)
        mask_image = mask_image.convert("RGB").resize((3384, 1710), Image.NEAREST)

        original_array = np.array(original_image)
        mask_array = np.array(mask_image)

        mask = np.all(mask_array == [255, 255, 255], axis=-1)
        original_array[mask] = [0, 255, 0]

        return Image.fromarray(original_array)

    @classmethod
    def process_image(cls, image: Image.Image, unet):
        detected_image = unet.detect_image(image)
        inferred_image = PreCA.infer(detected_image)
        smoothed_image = PreCA.ca_smooth(inferred_image)
        return cls.apply_mask(image, smoothed_image),smoothed_image


def main_page():
    st.title('自动驾驶车道线自动检测与增强')
    stframe = st.empty()
    st.sidebar.subheader("参数设置")

    is_pre = st.sidebar.checkbox('开启预测')
    unet = Unet(DEFAULTS) if is_pre else None

    if is_pre:
        u_ca_path = 'weights/best_conv_autoencoder1.pth'
        PreCA.initialize_model(u_ca_path)
    else:
        PreCA.unload_model()

    st.sidebar.subheader("图像检测")
    image_dir_path = st.sidebar.text_input('请输入图像文件夹路径:')
    is_get_iou = st.sidebar.checkbox('开启计算IOU')
    label_dir_path = st.sidebar.text_input('请输入标签文件夹路径:') if is_get_iou else None
    btn_click = st.sidebar.button("开始预测")

    if btn_click:
        process_images(image_dir_path, label_dir_path, unet, is_pre, is_get_iou, stframe)

    st.sidebar.subheader("视频检测")
    uploaded_video = st.sidebar.file_uploader("上传视频:", type=['mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'm4v'])

    if uploaded_video is not None:
        process_video(uploaded_video, unet, is_pre, stframe)


def process_images(image_dir_path, label_dir_path, unet, is_pre, is_get_iou, stframe):
    ious = []
    img_names = os.listdir(image_dir_path)
    iou_text = st.empty()

    for img_name in img_names:
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(image_dir_path, img_name)
            image = Image.open(image_path)
            if is_pre:
                result_image,smoothed_image = PreUnet.process_image(image, unet)
                stframe.image([image, result_image], width=640)
                if is_get_iou and label_dir_path:
                    label_path = os.path.join(label_dir_path, f"{os.path.splitext(img_name)[0]}_bin.png")
                    label = Image.open(label_path)
                    TP, FP, FN = PreUnet.calculate_metrics(smoothed_image, label)
                    iou = TP / (TP + FP + FN)
                    # ious.append(iou)
                    iou_text.text(f'当前IOU: {iou}')
            else:
                stframe.image(image, width=1024)


def process_video(uploaded_video, unet, is_pre, stframe):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    if 'frame_pos' not in st.session_state:
        st.session_state.frame_pos = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        if is_pre:
            processed_frame,smoothed_image = PreUnet.process_image(frame, unet)
            stframe.image(processed_frame, width=1024,use_column_width=False)
        else:
            stframe.image(frame, width=1024,use_column_width=False)

    cap.release()


if __name__ == '__main__':
    main_page()
